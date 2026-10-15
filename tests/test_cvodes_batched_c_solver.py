"""CVODES_batched_c: BE_batched_c's generated kernels and blocks, each block integrated by its own CVODES.

The models are test_batched_c_solver's small CellML files, a module array of unconnected cells written here, and
libcuflynx's 3compartment test model. Tests that build the library skip when SUNDIALS 6 or later is not found
(set SUNDIALS_DIR).
"""
import ctypes
import math
import re

import numpy as np
import pytest

from test_batched_c_solver import (CELLS, OPS, TRACED, _cell, _cellml, _cn, _ramp_protocol, _write, needs_c)

SOLVER = 'CVODES_batched_c'


def _skip_without_sundials(exc):
    if 'SUNDIALS' in str(exc):
        pytest.skip(f'{SOLVER} needs SUNDIALS 6 or later: {str(exc).splitlines()[-1]}')
    raise exc


def _helper(tmp_path, text, sim_time=1.0, dt=0.1, solver=SOLVER, name='model', **solver_info):
    from libcuflynx.generators.cpp.batched import BatchedBuildError
    from libcuflynx.solver_wrappers import get_simulation_helper
    solver_info.setdefault('build_dir', str(tmp_path / 'build'))
    try:
        return get_simulation_helper(_write(tmp_path, name, text), solver, 'cellml', dt=dt, sim_time=sim_time,
                                     solver_info=solver_info, pre_time=0.0)
    except BatchedBuildError as exc:
        _skip_without_sundials(exc)


def _module_array(n):
    """n unconnected cells, x' = -k x, each with its own k and x0: n blocks of one kernel."""
    cells = ''.join(_cell(f'cell_{i}', 0.1 + 0.05 * i, 1.0 + 0.1 * i) for i in range(n))
    env = '<component name="environment"><variable name="t" units="second" interface="public"/></component>'
    maps = ''.join(f'<connection component_1="environment" component_2="cell_{i}">'
                   f'<map_variables variable_1="t" variable_2="t"/></connection>' for i in range(n))
    return _cellml(cells + env + maps)


# cell_a's x starts at x0 = 4 k (a computed constant)
COMPUTED_X0 = CELLS.replace(
    '<variable name="x0" units="dimensionless" initial_value="2.0"/>',
    '<variable name="x0" units="dimensionless"/>').replace(
    '<apply><eq/><ci>flux</ci>',
    f'<apply><eq/><ci>x0</ci><apply><times/>{_cn(4, "second")}<ci>k</ci></apply></apply><apply><eq/><ci>flux</ci>', 1)


# ------------------------------------------------------------------ the schema and the generated C


@pytest.mark.unit
def test_cvodes_batched_c_is_a_cellml_solver_with_its_own_defaults():
    from libcuflynx.generators.cpp.batched import SOLVERS
    from libcuflynx.parsers.PrimitiveParsers import (SOLVER_INFO_FIELDS, SOLVER_SCHEMA, _solver_info_default_for,
                                                     validate_solver_info)
    from libcuflynx.solver_wrappers import batched_c_helper
    assert SOLVER in SOLVER_SCHEMA['solvers_by_model_type']['cellml']
    assert SOLVER_SCHEMA['methods_by_solver'][SOLVER] == list(SOLVERS[SOLVER]['methods']) == ['CVODE']
    assert SOLVER_SCHEMA['stiff_suitable_methods'][SOLVER] == ['CVODE']
    defaults = {f['name']: f['default'] for f in SOLVER_INFO_FIELDS[SOLVER]}
    assert defaults['rtol'] == batched_c_helper.CVODESSimulationHelper.DEFAULT_RTOL == 1e-6
    assert defaults['atol'] == batched_c_helper.CVODESSimulationHelper.DEFAULT_ATOL == 1e-8
    assert defaults['MaximumNumberOfSteps'] == batched_c_helper.DEFAULT_MAX_STEPS
    own = _solver_info_default_for('cellml', SOLVER)
    assert own['MaximumNumberOfSteps'] == 10_000_000 and 'MaximumStep' not in own and own['rtol'] == 1e-6
    validate_solver_info(SOLVER, {'rtol': 1e-6, 'threads': 4, 'jacobian': 'finite_difference', 'build_dir': '/x'})


def _exported(text):
    return set(re.findall(r'BK_API\s+[\w\s\*]*?\**(bk_\w+)\s*\(', text))


@pytest.mark.unit
def test_the_cvodes_runtime_exports_every_function_of_be_batched_c(tmp_path):
    """A caller can swap the libraries: the CVODES runtime exports BE_batched_c's functions (and one of its own,
    the CVODE flag of a failed block), and the kernels and libCellML's C are the same files."""
    from libcuflynx.generators import kernel_ir
    from libcuflynx.generators.cpp import batched
    _, am = kernel_ir.analyse_cellml(_write(tmp_path, 'cells', CELLS))
    km = kernel_ir.build_kernel_model(am)
    be, cv = (batched.sources(am, km, solver=s) for s in ('BE_batched_c', SOLVER))
    assert _exported(cv['batched_runtime.c']) - _exported(be['batched_runtime.c']) == {'bk_block_cvodes_flag'}
    assert _exported(be['batched_runtime.c']) <= _exported(cv['batched_runtime.c'])
    assert {'bk_instance_advance', 'bk_instances_advance', 'bk_set_trace', 'bk_solver_name',
            'bk_set_stop_at_outputs'} <= _exported(
        cv['batched_runtime.c'])
    for name in ('batched_kernels.c', 'batched_kernels.h', 'model0d_core.c', 'blocks.json'):
        assert be[name] == cv[name]
    assert 'CVodeReInit' in cv['batched_runtime.c'] and 'CVode' not in be['batched_runtime.c']
    assert 'SUNDIALS::cvodes' in cv['CMakeLists.txt'] and 'SUNDIALS' not in be['CMakeLists.txt']
    with pytest.raises(ValueError, match='solver must be one of'):
        batched.sources(am, km, solver='CVODE_myokit')


@pytest.mark.unit
def test_a_failed_cvodes_build_says_how_to_find_sundials(tmp_path, monkeypatch):
    from libcuflynx.coupling import model0d_lib
    from libcuflynx.generators import kernel_ir
    from libcuflynx.generators.cpp import batched

    def no_sundials(*args, **kwargs):
        raise model0d_lib.Model0dError('Could NOT find SUNDIALS_INCLUDE_DIR')
    monkeypatch.setattr(model0d_lib, 'build_library', no_sundials)
    _, am = kernel_ir.analyse_cellml(_write(tmp_path, 'cells', CELLS))
    km = kernel_ir.build_kernel_model(am)
    with pytest.raises(batched.BatchedBuildError, match='set SUNDIALS_DIR'):
        batched.build(batched.sources(am, km, solver=SOLVER), str(tmp_path))


# ------------------------------------------------------------------ solutions


@needs_c
@pytest.mark.solver
def test_solutions_converge_onto_the_analytic_ones(tmp_path):
    t = np.linspace(0.0, 3.0, 31)
    previous = None
    for rtol in (1e-4, 1e-6, 1e-8):
        helper = _helper(tmp_path, CELLS, sim_time=3.0, rtol=rtol, atol=1e-12)
        assert helper.library.solver == SOLVER
        assert helper.run()
        a, b, p = (helper.get_results([[n]])[0][0] for n in ('cell_a/x', 'cell_b/x', 'pair/p'))
        worst = max(np.max(np.abs(a - 2.0 * np.exp(-0.5 * t))) / 2.0, np.max(np.abs(b - np.exp(-0.2 * t))),
                    np.max(np.abs(p - np.cos(3.0 * t))))
        if previous is not None:
            assert worst < 0.3 * previous
        previous = worst
    assert worst < 1e-6
    flux = helper.get_results([['cell_a/flux']])[0][0]
    assert np.allclose(flux, 0.5 * a, rtol=1e-12)
    stats = helper.stats()
    assert stats['jacobians'] > 0 and stats['finite_difference_jacobians'] == 0


@needs_c
@pytest.mark.solver
def test_agrees_with_be_batched_c_and_cvode_myokit(tmp_path):
    from libcuflynx.solver_wrappers import get_simulation_helper
    path = _write(tmp_path, 'ops_model', OPS)
    names = [['ops/x', 'ops/y', 'ops/v']]
    out = {}
    for solver, info in (('CVODE_myokit', {'rtol': 1e-10, 'atol': 1e-12}),
                         ('BE_batched_c', {'rtol': 1e-8, 'atol': 1e-12, 'method': 'trbdf2'}),
                         (SOLVER, {'rtol': 1e-9, 'atol': 1e-12})):
        if solver != 'CVODE_myokit':
            info['build_dir'] = str(tmp_path / 'build')
        helper = (_helper(tmp_path, OPS, 4.0, 0.05, solver, 'ops_model', **info) if solver == SOLVER else
                  get_simulation_helper(path, solver, 'cellml', dt=0.05, sim_time=4.0, solver_info=info))
        assert helper.run()
        out[solver] = np.array(helper.get_results(names)[0])
    scale = np.max(np.abs(out['CVODE_myokit']), axis=1, keepdims=True)
    assert np.max(np.abs(out[SOLVER] - out['CVODE_myokit']) / scale) < 1e-6
    assert np.max(np.abs(out[SOLVER] - out['BE_batched_c']) / scale) < 1e-4


@needs_c
@pytest.mark.solver
def test_finite_difference_jacobians_are_cvodes_difference_quotients(tmp_path):
    exact = _helper(tmp_path, OPS, sim_time=4.0, dt=0.05, rtol=1e-9, atol=1e-12)
    fd = _helper(tmp_path, OPS, sim_time=4.0, dt=0.05, rtol=1e-9, atol=1e-12, jacobian='finite_difference')
    assert exact.run() and fd.run()
    x, x_fd = (h.get_results([['ops/x']])[0][0] for h in (exact, fd))
    assert np.max(np.abs(x - x_fd)) < 1e-6
    stats = fd.stats()
    assert stats['jacobians'] == 0 and stats['finite_difference_jacobians'] > 0


@needs_c
@pytest.mark.solver
@pytest.mark.slow
def test_agrees_with_be_batched_c_and_cvode_myokit_on_3compartment(generated_cellml_model_factory, tmp_path):
    """libcuflynx's 3compartment circulation (one block) over two heartbeats: the states, relative to each one's
    range (valve states switch, so their algebraic indicators can differ by a whole switch at an output)."""
    from libcuflynx.solver_wrappers import get_simulation_helper
    path = generated_cellml_model_factory('3compartment')
    names = None
    out = {}
    for solver, info in (('BE_batched_c', {'rtol': 1e-7, 'atol': 1e-10, 'method': 'trbdf2'}),
                         (SOLVER, {'rtol': 1e-8, 'atol': 1e-10}),
                         ('CVODE_myokit', {'rtol': 1e-9, 'atol': 1e-10, 'MaximumStep': 0.001})):
        if solver != 'CVODE_myokit':
            info['build_dir'] = str(tmp_path / 'build')
        try:
            helper = get_simulation_helper(path, solver, 'cellml', dt=0.01, sim_time=2.0, solver_info=info,
                                           pre_time=0.0)
        except Exception as exc:
            if solver != SOLVER:
                raise
            _skip_without_sundials(exc)
        if names is None:
            names = [n.replace('/', '.', 1) for n in helper.state_names]
        assert helper.run()
        out[solver] = np.array(helper.get_results(names, flatten=True))
    scale = np.maximum(np.max(np.abs(out['CVODE_myokit']), axis=1, keepdims=True), 1e-12)
    assert np.max(np.abs(out[SOLVER] - out['CVODE_myokit']) / scale) < 3e-4
    assert np.max(np.abs(out[SOLVER] - out['BE_batched_c']) / scale) < 3e-4


@needs_c
@pytest.mark.solver
def test_results_are_the_same_on_any_number_of_threads(tmp_path):
    results = []
    for threads in (1, 3, 8):
        helper = _helper(tmp_path, _module_array(16), sim_time=5.0, rtol=1e-6, threads=threads)
        assert helper.run()
        results.append(np.array(helper.get_all_results(flatten=True)))
    assert np.array_equal(results[0], results[1]) and np.array_equal(results[0], results[2])
    t = np.linspace(0.0, 5.0, 51)
    x15 = helper.get_results([['cell_15/x']])[0][0]
    assert np.max(np.abs(x15 - 2.5 * np.exp(-0.85 * t))) < 1e-4


@needs_c
@pytest.mark.solver
def test_a_block_whose_inputs_did_not_change_continues_its_integration(tmp_path):
    """Changing cell_b's rate constant between sub-experiments reinitialises cell_b's CVODES only: cell_a's outputs
    are the same bits as in one unbroken run, and cell_b follows the new constant from where it was."""
    from libcuflynx.protocol_runners.protocol_executor import ProtocolExecutor
    whole = _helper(tmp_path, CELLS, sim_time=4.0, dt=0.5, rtol=1e-8, atol=1e-12)
    assert whole.run()
    unbroken = whole.get_results([['cell_a/x']])[0][0]
    helper = _helper(tmp_path, CELLS, sim_time=2.0, dt=0.5, rtol=1e-8, atol=1e-12)
    protocol = {'pre_times': [0.0], 'sim_times': [[2.0, 2.0]], 'params_to_change': {'cell_b/k': [[0.2, 1.0]]}}
    ok, by_sub, _, t_by_exp = ProtocolExecutor(helper).run_protocol(protocol, result_variables=[['cell_a/x',
                                                                                                   'cell_b/x']])
    assert ok
    a = np.concatenate([by_sub[(0, 0)][0][0], by_sub[(0, 1)][0][0][1:]])
    b = np.concatenate([by_sub[(0, 0)][0][1], by_sub[(0, 1)][0][1][1:]])
    assert np.array_equal(a, unbroken)
    t = t_by_exp[0]
    expected = np.where(t <= 2.0, np.exp(-0.2 * t), np.exp(-0.4) * np.exp(-1.0 * (t - 2.0)))
    assert np.max(np.abs(b - expected)) < 1e-6


@needs_c
@pytest.mark.solver
def test_a_state_initialised_by_a_parameter_follows_it(tmp_path):
    helper = _helper(tmp_path, COMPUTED_X0, sim_time=1.0, rtol=1e-8, atol=1e-12)
    helper.set_param_vals(['cell_a/k'], [1.0])
    assert helper.run()
    x = helper.get_results([['cell_a/x']])[0][0]
    assert x[0] == 4.0 and x[-1] == pytest.approx(4.0 * math.exp(-1.0), rel=1e-6)
    helper.set_param_vals(['cell_a/k'], [0.25], change_states=False)
    helper.update_times(helper.dt, 1.0, 1.0, 0.0)
    assert helper.run()
    y = helper.get_results([['cell_a/x']])[0][0]
    assert y[0] == pytest.approx(x[-1], rel=1e-12) and y[-1] == pytest.approx(x[-1] * math.exp(-0.25), rel=1e-6)


@needs_c
@pytest.mark.solver
def test_too_many_steps_fails_the_run(tmp_path):
    helper = _helper(tmp_path, CELLS, sim_time=3.0, rtol=1e-8, MaximumNumberOfSteps=20)
    with pytest.warns(UserWarning, match='too many steps'):
        assert not helper.run()
    helper.max_steps = 10_000_000
    helper.sim.set_options(rtol=1e-8, atol=1e-8, max_steps=10_000_000, newton_tol=0.0, max_newton=0,
                           max_jacobian_age=0)
    assert helper.run()


# ------------------------------------------------------------------ protocol traces


@needs_c
@pytest.mark.solver
def test_a_constant_follows_its_protocol_trace(tmp_path):
    """x' = u with u a ramp to 2 over 0.5 s then held: x(t) = 2 t^2 to 0.5 s, then 0.5 + 2 (t - 0.5). The
    breakpoint is a stop time, so no step crosses the kink."""
    helper = _helper(tmp_path, TRACED, sim_time=1.0, dt=0.05, rtol=1e-8, atol=1e-12)
    helper.set_protocol_info(_ramp_protocol())
    helper.set_param_vals(['drive/u'], ['ramp'])
    assert helper.run()
    t = helper.get_time()
    x, flux, u = helper.get_results([['drive/x', 'drive/flux', 'drive/u']])[0]
    assert x == pytest.approx(np.where(t <= 0.5, 2 * t ** 2, 0.5 + 2 * (t - 0.5)), abs=1e-7)
    assert flux == pytest.approx(np.interp(t, [0.0, 0.5], [0.0, 2.0]))
    assert u == pytest.approx(flux)
    helper.set_param_vals(['drive/u'], [0.0], change_states=False)     # a number again ends the trace
    helper.update_times(helper.dt, 1.0, 1.0, 0.0)
    assert helper.run()
    x, u = helper.get_results([['drive/x', 'drive/u']])[0]
    assert x == pytest.approx(np.full(len(t), x[-1])) and np.all(u == 0.0)


@needs_c
@pytest.mark.solver
def test_a_trace_runs_through_the_protocol_executor_as_on_the_other_solvers(tmp_path):
    from libcuflynx.protocol_runners.protocol_executor import ProtocolExecutor
    from libcuflynx.solver_wrappers import get_simulation_helper
    path = _write(tmp_path, 'model', TRACED)
    out = {}
    for solver, info in [(SOLVER, {'rtol': 1e-8, 'atol': 1e-12}),
                         ('BE_batched_c', {'rtol': 1e-8, 'atol': 1e-12, 'method': 'trbdf2'}),
                         ('CVODE_myokit', {'rtol': 1e-10, 'atol': 1e-12, 'MaximumStep': 0.01})]:
        if solver != 'CVODE_myokit':
            info['build_dir'] = str(tmp_path / 'build')
        helper = (_helper(tmp_path, TRACED, 1.0, 0.05, **info) if solver == SOLVER else
                  get_simulation_helper(path, solver, 'cellml', dt=0.05, sim_time=1.0, solver_info=info, pre_time=0.0))
        helper.set_protocol_info(_ramp_protocol())
        ok, by_sub, _, _ = ProtocolExecutor(helper).run_protocol(_ramp_protocol(), result_variables=[['drive/x']])
        assert ok
        out[solver] = np.asarray(by_sub[(0, 0)][0][0] if len(by_sub[(0, 0)]) == 1 else by_sub[(0, 0)][0],
                                 dtype=float).ravel()
    assert out[SOLVER] == pytest.approx(out['CVODE_myokit'], abs=1e-6)
    assert out[SOLVER] == pytest.approx(out['BE_batched_c'], abs=1e-6)


# ------------------------------------------------------------------ the instance API


class _Instances:
    """The instance API of a built CVODES_batched_c (or BE_batched_c) library, through ctypes."""

    def __init__(self, tmp_path, text, outputs=(), solver=SOLVER):
        from libcuflynx.generators.cpp import batched
        self.solver = solver
        try:
            _, path, _, _ = batched.load_or_build(_write(tmp_path, 'model', text), str(tmp_path / 'build'),
                                                  outputs=outputs, solver=solver)
            lib = batched.BatchedLibrary(path)._lib
        except batched.BatchedBuildError as exc:
            _skip_without_sundials(exc)
        vp, d, i, s = ctypes.c_void_p, ctypes.c_double, ctypes.c_int, ctypes.c_char_p
        dp = ctypes.POINTER(d)
        for name, res, args in [
                ('bk_solver_name', s, []),
                ('bk_instance_create', vp, []), ('bk_instance_copy', vp, [vp]), ('bk_instance_destroy', None, [vp]),
                ('bk_instance_states', dp, [vp]), ('bk_instance_variables', dp, [vp]),
                ('bk_instance_time', d, [vp]), ('bk_instance_sim', vp, [vp]), ('bk_instance_restart', None, [vp]),
                ('bk_instance_parameters_changed', None, [vp]), ('bk_instance_advance', i, [vp, d]),
                ('bk_instances_advance', i, [i, ctypes.POINTER(vp), d, i]),
                ('bk_instance_compute_variables', None, [vp]), ('bk_state_count', i, []), ('bk_variable_count', i, []),
                ('bk_instance_outputs', None, [vp]), ('bk_output_count', i, []), ('bk_output_index', i, [i]),
                ('bk_instance_create_blocks', vp, [i, ctypes.POINTER(i)]), ('bk_state_block', i, [i]),
                ('bk_instance_initialise_states', None, [vp]), ('bk_stats', None, [vp, ctypes.POINTER(ctypes.c_long)]),
                ('bk_state_info', None, [i] + [ctypes.POINTER(s)] * 3),
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

    def create(self, rtol=1e-9, blocks=None):
        if blocks is None:
            inst = self.lib.bk_instance_create()
        else:
            inst = self.lib.bk_instance_create_blocks(len(blocks), (ctypes.c_int * len(blocks))(*blocks))
        newton = (0.0, 0, 0) if self.solver == SOLVER else (0.01, 6, 50)     # CVODES's defaults; BE_batched_c's
        self.lib.bk_set_options(self.lib.bk_instance_sim(inst), rtol, 1e-12, 0.0, 0.0, *newton, 10_000_000, 0, 0)
        return inst

    def states(self, inst):
        return np.ctypeslib.as_array(self.lib.bk_instance_states(inst), shape=(self.lib.bk_state_count(),))

    def variables(self, inst):
        return np.ctypeslib.as_array(self.lib.bk_instance_variables(inst), shape=(self.lib.bk_variable_count(),))

    def steps(self, inst):
        out = (ctypes.c_long * 7)()
        self.lib.bk_stats(self.lib.bk_instance_sim(inst), out)
        return out[0]


@needs_c
@pytest.mark.solver
def test_an_instance_stepped_by_its_caller_follows_the_analytic_solutions(tmp_path):
    api = _Instances(tmp_path, CELLS)
    assert api.lib.bk_solver_name() == SOLVER.encode()
    inst = api.create()
    for _ in range(10):
        assert api.lib.bk_instance_advance(inst, 0.1) == 0
    y = api.states(inst)
    assert api.lib.bk_instance_time(inst) == pytest.approx(1.0)
    assert y[api.state['cell_a/x']] == pytest.approx(2.0 * math.exp(-0.5), rel=1e-7)
    assert y[api.state['pair/p']] == pytest.approx(math.cos(3.0), rel=1e-6)
    api.lib.bk_instance_destroy(inst)


@needs_c
@pytest.mark.solver
def test_a_changed_parameter_reinitialises_only_the_blocks_that_read_it(tmp_path):
    """Setting cell_a's k at t = 1 reinitialises cell_a's CVODES (x keeps its value, then follows the new k); cell_b
    and the oscillator read nothing that changed, so they continue: the same bits as an instance left alone."""
    api = _Instances(tmp_path, COMPUTED_X0)
    changed, alone = api.create(), api.create()
    x, k, x0 = api.state['cell_a/x'], api.variable['cell_a/k'], api.variable['cell_a/x0']
    for _ in range(10):
        api.lib.bk_instance_advance(changed, 0.1)
        api.lib.bk_instance_advance(alone, 0.1)
    x1 = api.states(changed)[x]
    api.variables(changed)[k] = 0.25
    api.lib.bk_instance_parameters_changed(changed)
    assert api.variables(changed)[x0] == 1.0 and api.states(changed)[x] == x1
    for _ in range(10):
        api.lib.bk_instance_advance(changed, 0.1)
        api.lib.bk_instance_advance(alone, 0.1)
    assert api.states(changed)[x] == pytest.approx(x1 * math.exp(-0.25), rel=1e-7)
    for name in ('cell_b/x', 'pair/p', 'pair/q'):
        assert api.states(changed)[api.state[name]] == api.states(alone)[api.state[name]]
    assert api.states(changed)[x] != api.states(alone)[x]
    api.lib.bk_instance_destroy(changed)
    api.lib.bk_instance_destroy(alone)


@needs_c
@pytest.mark.solver
def test_set_states_restart_the_integration_from_them(tmp_path):
    api = _Instances(tmp_path, CELLS)
    inst = api.create()
    for _ in range(5):
        api.lib.bk_instance_advance(inst, 0.1)
    x = api.state['cell_a/x']
    api.states(inst)[x] = 3.0
    api.lib.bk_instance_restart(inst)
    for _ in range(5):
        api.lib.bk_instance_advance(inst, 0.1)
    assert api.states(inst)[x] == pytest.approx(3.0 * math.exp(-0.25), rel=1e-7)
    api.lib.bk_instance_destroy(inst)


@needs_c
@pytest.mark.solver
@pytest.mark.parametrize('solver', [SOLVER, 'BE_batched_c'])
def test_instances_advanced_on_threads_match_one_thread_bit_for_bit(tmp_path, solver):
    """Sixteen instances with different rate constants, advanced together by bk_instances_advance on 1 and 8
    OpenMP threads, and one by one from Python threads."""
    from concurrent.futures import ThreadPoolExecutor
    api = _Instances(tmp_path, CELLS, solver=solver)
    k = api.variable['cell_a/k']

    def run(threads, python_threads=False):
        instances = [api.create(rtol=1e-6) for _ in range(16)]
        for j, inst in enumerate(instances):
            api.variables(inst)[k] = 0.1 * (j + 1)
            api.lib.bk_instance_parameters_changed(inst)
        array = (ctypes.c_void_p * len(instances))(*instances)
        if python_threads:
            def step(inst):
                for _ in range(50):
                    api.lib.bk_instance_advance(inst, 0.1)
            with ThreadPoolExecutor(threads) as pool:
                list(pool.map(step, instances))
        else:
            for _ in range(50):
                assert api.lib.bk_instances_advance(len(instances), array, 0.1, threads) == 0
        out = np.array([api.states(inst).copy() for inst in instances])
        for inst in instances:
            api.lib.bk_instance_destroy(inst)
        return out
    one = run(1)
    assert np.array_equal(one, run(8)) and np.array_equal(one, run(8, python_threads=True))
    assert len(set(one[:, api.state['cell_a/x']])) == 16


@needs_c
@pytest.mark.solver
def test_a_copy_follows_its_original(tmp_path):
    """A copy at t = 0.5 starts CVODES afresh from the copied states: it agrees with its original to the
    tolerance, and the original continues undisturbed."""
    api = _Instances(tmp_path, CELLS)
    inst, alone = api.create(), api.create()
    for _ in range(5):
        api.lib.bk_instance_advance(inst, 0.1)
        api.lib.bk_instance_advance(alone, 0.1)
    copy = api.lib.bk_instance_copy(inst)
    assert api.lib.bk_instance_time(copy) == api.lib.bk_instance_time(inst)
    assert np.array_equal(api.states(copy), api.states(inst))
    for _ in range(5):
        for one in (inst, copy, alone):
            api.lib.bk_instance_advance(one, 0.1)
    assert np.allclose(api.states(copy), api.states(inst), rtol=1e-7, atol=1e-9)
    assert np.array_equal(api.states(inst), api.states(alone))
    for one in (inst, copy, alone):
        api.lib.bk_instance_destroy(one)


@needs_c
@pytest.mark.solver
def test_an_instance_of_some_blocks_steps_only_those(tmp_path):
    api = _Instances(tmp_path, CELLS)
    b = api.lib.bk_state_block(api.state['cell_a/x'])
    inst = api.create(blocks=[b])
    for _ in range(10):
        api.lib.bk_instance_advance(inst, 0.1)
    y = api.states(inst)
    assert y[api.state['cell_a/x']] == pytest.approx(2.0 * math.exp(-0.5), rel=1e-7)
    assert y[api.state['cell_b/x']] == 1.0 and y[api.state['pair/p']] == 1.0
    api.lib.bk_instance_destroy(inst)


@needs_c
@pytest.mark.solver
def test_outputs_and_initial_values_as_on_be_batched_c(tmp_path):
    api = _Instances(tmp_path, CELLS, outputs=('cell_b/flux',))
    flux_b, flux_a = api.variable['cell_b/flux'], api.variable['cell_a/flux']
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
    v[api.variable['cell_a/x0']] = 5.0
    api.lib.bk_instance_initialise_states(inst)
    assert api.states(inst)[api.state['cell_a/x']] == 5.0
    api.lib.bk_instance_advance(inst, 0.2)
    assert api.states(inst)[api.state['cell_a/x']] == pytest.approx(5.0 * math.exp(-0.1), rel=1e-7)
    api.lib.bk_instance_destroy(inst)


@pytest.mark.unit
def test_a_user_inputs_config_selects_it(base_user_inputs, temp_output_dir, temp_generated_models_dir):
    """solver: CVODES_batched_c in user_inputs passes validation with its method (CVODE) and its own step limit."""
    from libcuflynx.parsers.PrimitiveParsers import YamlFileParser
    config = dict(base_user_inputs, file_prefix='3compartment', input_param_file='3compartment_parameters.csv',
                  model_type='cellml', solver=SOLVER, solver_info={'rtol': 1e-7, 'threads': 4},
                  generated_models_dir=temp_generated_models_dir, param_id_output_dir=temp_output_dir)
    parsed = YamlFileParser().parse_user_inputs_file(config, obs_path_needed=False)
    info = parsed['solver_info']
    assert info['solver'] == SOLVER and info['method'] == 'CVODE'
    assert info['rtol'] == 1e-7 and info['threads'] == 4 and info['MaximumNumberOfSteps'] == 10_000_000


@needs_c
@pytest.mark.solver
def test_stop_at_outputs_is_accepted_and_changes_nothing(tmp_path):
    """BE_batched_c's stop_at_outputs is a no-op here (CVODES steps past outputs and interpolates), so callers can
    set it on either library."""
    results = []
    for stop in (False, True):
        helper = _helper(tmp_path, CELLS, rtol=1e-6)
        helper.sim.set_stop_at_outputs(stop)
        for j in range(10):
            assert helper.sim.run(helper.states, helper.variables, 0.1 * j, [0.1 * (j + 1)]) == []
        results.append(helper.states.copy())
    assert np.array_equal(results[0], results[1])
