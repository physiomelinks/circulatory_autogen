"""Every forward backend against one converged reference, on small, event and large models.

The reference is CVODE_myokit at rtol 1e-11, checked against rtol 1e-10 before use
(tests/_reference_solutions.py). Each backend is compared with it by three metrics
(tests/_solver_metrics.py), gated by that module's TOLERANCES table:

    scaled_rms          smooth accuracy
    shift_tolerant_max  accuracy that a valve opening one output step early does not fail
    feature_error       the mean/min/max a calibration reads

The MATRIX lists which backend runs which model. A pair that cannot run is in EXCLUDED with
its reason, rather than appearing as a skip, and test_every_solver_in_schema_has_a_matrix_row
fails when SOLVER_SCHEMA gains a solver or method this file does not account for.
"""
import contextlib
import io
import json
import os
import shutil
import subprocess

import numpy as np
import pytest

from _reference_solutions import reference_solution, run_states, state_names
from _solver_metrics import (ROUND_TRIP, SHIFT_STEPS, TOLERANCES, convergence_order,
                             feature_error, scaled_rms, shift_tolerant_max, worst)

pytestmark = [pytest.mark.integration, pytest.mark.solver]

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESOURCES = os.path.join(REPO, 'resources')
GOLDEN = os.path.join(REPO, 'tests', 'data', 'references', 'Lotka_Volterra.npz')

# name: (sim_time, dt, reference atol). Every run starts at the parameter file's initial
# state (pre_time 0), so the large models stay at 1 s. The reference atol is per model: Myokit
# takes one absolute tolerance for every state, and physiological's smallest volumes need
# 1e-14 to converge, while at 1e-15 control_parasymp's mitral valve, which sits close to its
# switching point, opens on a different step than at every looser setting.
MODELS = {
    'Lotka_Volterra': (5.0, 0.01, 1e-12),
    '3compartment': (2.0, 0.001, 1e-12),
    'SN_simple': (1.0, 0.001, 1e-12),
    'physiological': (1.0, 0.001, 1e-14),
    'cvs_model_0d': (1.0, 0.001, 1e-14),
    'control_parasymp': (1.0, 0.001, 1e-12),
}
SMALL = ['Lotka_Volterra', '3compartment', 'SN_simple']
LARGE = ['physiological', 'cvs_model_0d', 'control_parasymp']


def _solver_info(backend, dt):
    if backend in ('CVODE_myokit', 'CVODE_libopencor'):
        return {'rtol': 1e-8, 'atol': 1e-10, 'MaximumStep': dt, 'MaximumNumberOfSteps': 10 ** 6}
    if backend == 'BE_batched_c':
        return {'method': 'trbdf2', 'rtol': 1e-6}
    if backend == 'CVODES_batched_c':
        return {'rtol': 1e-8, 'atol': 1e-10, 'MaximumStep': dt}
    if backend == 'casadi_integrator':
        return {'method': 'cvodes', 'reltol': 1e-8, 'abstol': 1e-10, 'max_step_size': dt,
                'max_num_steps': 100000}
    if backend == 'solve_ivp':
        return {'method': 'BDF', 'rtol': 1e-8, 'atol': 1e-10, 'max_step': dt}
    if backend == 'aadc_semi_implicit':
        # semi_implicit (the solver's name) is first order with diagonal damping: 0.26 of the
        # range off on Lotka-Volterra at dt 0.01. adaptive_rk45 is explicit and takes minutes on
        # 3compartment. bdf_newton at max_step 1e-4 takes seconds and is within 3e-4 rms.
        return {'method': 'bdf_newton', 'max_step': 1e-4}
    raise ValueError(backend)


MODEL_TYPE = {'CVODE_myokit': 'cellml', 'CVODE_libopencor': 'cellml', 'BE_batched_c': 'cellml',
              'CVODES_batched_c': 'cellml', 'casadi_integrator': 'casadi_python',
              'solve_ivp': 'python', 'aadc_semi_implicit': 'aadc_python'}

# backend: the models it runs here. Every (backend, model) pair outside this is in EXCLUDED.
MATRIX = {
    'CVODE_myokit': SMALL + LARGE,
    'CVODE_libopencor': SMALL + LARGE,
    'BE_batched_c': SMALL + LARGE,
    'CVODES_batched_c': SMALL + LARGE,
    'casadi_integrator': SMALL + LARGE,
    'solve_ivp': ['Lotka_Volterra'],
    'aadc_semi_implicit': ['Lotka_Volterra', '3compartment'],
}
EXCLUDED = {
    ('solve_ivp', m): 'scipy BDF with a Python right-hand side; minutes on this model, and the '
                      'SN models fail the libCellML analyser (see test_solvers.py)'
    for m in ['3compartment', 'SN_simple'] + LARGE}
EXCLUDED.update({
    ('aadc_semi_implicit', m): 'AADC (release checks only) is checked on the two small models'
    for m in ['SN_simple'] + LARGE})

# Solvers and methods of SOLVER_SCHEMA that are covered by another test, not by the matrix.
# Each names that test. test_every_solver_in_schema_has_a_matrix_row reads this.
COVERED_ELSEWHERE = {
    'CVODE_opencor': 'removed by #638 (CVODE_libopencor replaces it)',
    'CVODE': 'test_cpp_cvode_matches_the_reference (model_type cpp)',
    'RK4': 'test_cpp_template_generator.py (fixed-step cpp builds)',
    'explEul': 'test_cpp_template_generator.py (fixed-step cpp builds)',
    'Heun': 'test_cpp_template_generator.py (fixed-step cpp builds)',
    'midpoint': 'test_cpp_template_generator.py (fixed-step cpp builds)',
    'external': 'test_external_simulation_helper.py (the user owns the scheme)',
}
COVERED_METHODS ={('BE_batched_c', 'backward_euler'): 'test_batched_c_solver.py'}
COVERED_METHODS.update({('solve_ivp', m): 'scipy integrates; the matrix checks BDF'
                        for m in ['RK45', 'RK23', 'DOP853', 'Radau', 'LSODA', 'forward_euler']})
COVERED_METHODS.update({('casadi_integrator', m): 'test_solvers.py / test_casadi_*.py'
                        for m in ['idas', 'collocation', 'rk']})
COVERED_METHODS.update({('casadi_integrator', m): 'test_fixed_step_convergence_order'
                        for m in ['bdf', 'semi_implicit_euler']})
COVERED_METHODS.update({('aadc_semi_implicit', m): 'test_aadc_solvers.py (release checks)'
                        for m in ['adaptive_rk45', 'semi_implicit', 'semi_implicit_signed',
                                  'implicit_euler_ift', 'implicit_newton', 'rk4']})


def _skip_unless_available(backend):
    if backend == 'CVODE_libopencor':
        from libcuflynx.parsers.PrimitiveParsers import SOLVER_SCHEMA
        if backend not in SOLVER_SCHEMA['solvers_by_model_type']['cellml']:
            pytest.skip('CVODE_libopencor is not a solver here yet (it arrives with #638)')
        pytest.importorskip('libopencor')
    elif backend == 'casadi_integrator':
        pytest.importorskip('casadi')
    elif backend == 'aadc_semi_implicit':
        pytest.importorskip('aadc')
    elif backend in ('BE_batched_c', 'CVODES_batched_c'):
        if shutil.which('cmake') is None:
            pytest.skip('cmake not found, which the batched C solvers need')


# --------------------------------------------------------------------------- models


class _Models:
    """Generates each model once per session: CellML, then the Python model types from it."""

    def __init__(self, folder):
        self.folder = folder
        self.paths = {}

    def cellml(self, name):
        if name not in self.paths:
            from libcuflynx.scripts.script_generate_with_new_architecture import \
                generate_with_new_architecture
            from libcuflynx.utilities.utility_funcs import get_default_inp_data_dict
            inp = get_default_inp_data_dict(name, f'{name}_parameters.json', RESOURCES)
            inp.update(generated_models_dir=self.folder, DEBUG=True, model_type='cellml')
            with contextlib.redirect_stdout(io.StringIO()):
                assert generate_with_new_architecture(False, inp), f'generating {name} failed'
            self.paths[name] = os.path.join(self.folder, name, f'{name}.cellml')
        return self.paths[name]

    def python(self, name, flavour):
        key = (name, flavour)
        if key not in self.paths:
            from libcuflynx.generators.PythonGenerator import PythonGenerator
            out = os.path.join(self.folder, f'{name}_{flavour}')
            kwargs = {'casadi_compat': flavour == 'casadi', 'aadc_compat': flavour == 'aadc'}
            with contextlib.redirect_stdout(io.StringIO()):
                self.paths[key] = PythonGenerator(self.cellml(name), output_dir=out,
                                                  module_name=name, **kwargs).generate()
        return self.paths[key]

    def path_for(self, backend, name):
        model_type = MODEL_TYPE[backend]
        if model_type == 'cellml':
            return self.cellml(name)
        return self.python(name, {'casadi_python': 'casadi', 'python': 'python',
                                  'aadc_python': 'aadc'}[model_type])


@pytest.fixture(scope='module')
def models(tmp_path_factory):
    return _Models(str(tmp_path_factory.mktemp('solver_matrix_models')))


def _helper(models, backend, name, sim_time=None, dt=None, solver_info=None):
    from libcuflynx.solver_wrappers import get_simulation_helper
    default_sim_time, default_dt, _ = MODELS[name]
    dt = dt or default_dt
    with contextlib.redirect_stdout(io.StringIO()):
        return get_simulation_helper(
            model_path=models.path_for(backend, name), solver=backend,
            model_type=MODEL_TYPE[backend], dt=dt, sim_time=sim_time or default_sim_time,
            solver_info=solver_info or _solver_info(backend, dt), pre_time=0.0)


def _close(helper):
    if hasattr(helper, 'close_simulation'):
        helper.close_simulation()


def _reference(models, name):
    sim_time, dt, atol = MODELS[name]
    return reference_solution(models.cellml(name), sim_time, dt, atol)


def _assert_close(x, ref, names, gates, label):
    report = {metric: worst(f(x, ref), names) for metric, f in
              (('scaled_rms', scaled_rms), ('shift_max', shift_tolerant_max),
               ('features', feature_error))}
    failed = {m: v for m, v in report.items() if v[0] > gates[m]}
    assert not failed, (f'{label}: ' + '; '.join(
        f'{m} {v[0]:.2e} in {v[1]} (gate {gates[m]:.0e})' for m, v in failed.items()))


# --------------------------------------------------------------------------- the metrics


@pytest.mark.unit
def test_a_one_step_shift_of_a_step_passes_shift_tolerant_max_and_fails_a_plain_max():
    """Why shift_tolerant_max exists: a valve opening one output step late is not an error."""
    t = np.arange(100)
    ref = (t >= 50).astype(float)[None, :]
    late = (t >= 51).astype(float)[None, :]
    assert np.max(np.abs(late - ref)) == 1.0
    assert shift_tolerant_max(late, ref)[0] == 0.0
    too_late = (t >= 50 + SHIFT_STEPS + 1).astype(float)[None, :]
    assert shift_tolerant_max(too_late, ref)[0] == 1.0


@pytest.mark.unit
def test_convergence_order_reads_the_slope():
    steps = np.array([0.1, 0.05, 0.025])
    assert convergence_order(3 * steps ** 2, steps) == pytest.approx(2.0)


# --------------------------------------------------------------------------- the reference


def test_the_lotka_volterra_reference_matches_the_committed_golden(models):
    """A reference that drifts with a Myokit or SUNDIALS release shows up here first.

    The golden file is the converged reference from Myokit 1.39.1 with SUNDIALS 7.7.0.
    Regenerate it with `python tests/test_solver_matrix.py --write-golden` only when the
    model itself changes.
    """
    names, states = _reference(models, 'Lotka_Volterra')
    golden = np.load(GOLDEN)
    assert list(golden['names']) == names
    error, state = worst(scaled_rms(states, golden['states']), names)
    assert error < 1e-8, f'{state} moved by {error:.2e} from the golden reference'


# --------------------------------------------------------------------------- forward runs

_PAIRS = [(b, m) for b, ms in MATRIX.items() for m in ms]


@pytest.mark.parametrize('backend, model', _PAIRS, ids=[f'{b}-{m}' for b, m in _PAIRS])
def test_forward_matches_converged_reference(backend, model, models, record_property):
    record_property('backend', backend)
    _skip_unless_available(backend)
    names, ref = _reference(models, model)
    helper = _helper(models, backend, model)
    try:
        x = run_states(helper, names)
    finally:
        _close(helper)
    _assert_close(x, ref, names, TOLERANCES[backend], f'{backend} on {model}')


# --------------------------------------------------------------------------- fixed-step order


@pytest.mark.parametrize('method, order', [('bdf', 2), ('semi_implicit_euler', 1)])
def test_fixed_step_convergence_order(method, order, models, record_property):
    """casadi's fixed-step methods converge at their order on Lotka-Volterra."""
    record_property('backend', 'casadi_integrator')
    _skip_unless_available('casadi_integrator')
    names, ref = _reference(models, 'Lotka_Volterra')  # on a 0.01 grid
    steps, errors = [0.01, 0.005, 0.0025], []
    for h in steps:
        if method == 'bdf':
            dt, info = 0.01, {'method': 'bdf', 'max_step': h, 'max_num_steps': 10 ** 6}
        else:
            dt, info = h, {'method': 'semi_implicit_euler'}
        helper = _helper(models, 'casadi_integrator', 'Lotka_Volterra', dt=dt, solver_info=info)
        x = run_states(helper, names)[:, ::int(round(0.01 / dt))]
        errors.append(float(np.max(np.abs(x - ref))))
    assert convergence_order(errors, steps) == pytest.approx(order, abs=0.3), errors


# --------------------------------------------------------------------------- state between runs

_ROUND_TRIP_BACKENDS = ['CVODE_myokit', 'CVODE_libopencor', 'BE_batched_c', 'CVODES_batched_c',
                        'casadi_integrator', 'solve_ivp']
_LV_PARAMS = ['Lotka_Volterra/alpha', 'Lotka_Volterra/gamma']


@pytest.mark.parametrize('backend', _ROUND_TRIP_BACKENDS)
def test_parameter_round_trip_is_bit_identical(backend, models, record_property):
    """Run at A, at B, then at A again: the two A runs are the same.

    Every helper keeps state between runs: compiled simulations, cached integrators (casadi's
    process-wide integrator cache is keyed by model, grid and options), logged results. A
    parameter that reaches only some of it shows up as A != A.
    """
    record_property('backend', backend)
    _skip_unless_available(backend)
    names = state_names(models.cellml('Lotka_Volterra'))
    helper = _helper(models, backend, 'Lotka_Volterra')
    try:
        a = helper.get_init_param_vals(_LV_PARAMS)
        a = [float(np.ravel(v)[0]) for v in a]
        first = run_states(helper, names)
        # reset, then set: the order ProtocolExecutor uses. The Python helpers' reset restores
        # the model's own parameter values.
        helper.reset_and_clear()
        helper.set_param_vals(_LV_PARAMS, [1.3 * v for v in a])
        moved = run_states(helper, names)
        helper.reset_and_clear()
        helper.set_param_vals(_LV_PARAMS, a)
        again = run_states(helper, names)
    finally:
        _close(helper)
    assert np.max(np.abs(moved - first)) > 1e-3 * np.max(np.abs(first)), \
        'parameter B changed nothing, so this test proves nothing'
    np.testing.assert_allclose(again, first, rtol=ROUND_TRIP, atol=0)


@pytest.mark.parametrize('backend', _ROUND_TRIP_BACKENDS)
def test_state_reset_between_runs(backend, models, record_property):
    """reset_and_clear then run equals a fresh helper's run, on a model with valves."""
    record_property('backend', backend)
    _skip_unless_available(backend)
    model = 'Lotka_Volterra' if backend == 'solve_ivp' else '3compartment'
    names = state_names(models.cellml(model))
    used = _helper(models, backend, model, sim_time=0.5)
    fresh = _helper(models, backend, model, sim_time=0.5)
    try:
        run_states(used, names)
        used.reset_and_clear()
        again = run_states(used, names)
        expected = run_states(fresh, names)
    finally:
        _close(used)
        _close(fresh)
    np.testing.assert_allclose(again, expected, rtol=ROUND_TRIP, atol=0)


# --------------------------------------------------------------------------- what calibrations read


# The resources' obs_data spin each experiment up for pre_time first (3compartment: 20 s).
# The comparisons here are of one run at the same parameters, so a short pre_time is enough.
_PRE_TIME = 1.0


def _obs_path(tmp_path, model):
    with open(os.path.join(RESOURCES, f'{model}_obs_data.json')) as f:
        obs = json.load(f)
    pre_times = obs['protocol_info']['pre_times']
    obs['protocol_info']['pre_times'] = [min(p, _PRE_TIME) for p in pre_times]
    path = os.path.join(str(tmp_path), f'{model}_obs_data.json')
    with open(path, 'w') as f:
        json.dump(obs, f)
    return path


def _engine(tmp_path, models, backend, model, do_ad=False, solver_info=None):
    from libcuflynx.param_id.paramID import CVS0DParamID
    out = os.path.join(str(tmp_path), f'{backend}_{model}_{do_ad}')
    os.makedirs(out, exist_ok=True)
    info = dict(solver_info or {'solver': backend})
    info.setdefault('solver', backend)
    with contextlib.redirect_stdout(io.StringIO()):
        outer = CVS0DParamID(
            model_path=models.path_for(backend, model), model_type=MODEL_TYPE[backend],
            param_id_method='sp_minimize', file_name_prefix=model,
            params_for_id_path=os.path.join(RESOURCES, f'{model}_params_for_id.json'),
            param_id_obs_path=_obs_path(out, model),
            sim_time=None, pre_time=None, dt=MODELS[model][1], solver_info=info,
            do_ad=do_ad, DEBUG=True, param_id_output_dir=out)
    engine = outer.param_id
    engine.output_dir = out
    return engine


def _x0(engine):
    """The calibration's starting point: the model's values of the calibrated parameters."""
    raw = engine.sim_helper.get_init_param_vals(engine.param_id_info['param_names'])
    return np.array([np.ravel(v)[0] for v in raw], dtype=float)


_COST_BACKENDS = ['CVODE_libopencor', 'BE_batched_c', 'CVODES_batched_c', 'casadi_integrator']
_COST_TOLERANCE = {'BE_batched_c': 1e-2}  # relative; every other backend is held to 1e-3


@pytest.mark.parametrize('backend', _COST_BACKENDS)
def test_cost_and_features_agree_across_backends(backend, models, tmp_path, record_property):
    """3compartment's calibration cost and features from each backend equal CVODE_myokit's.

    This is what a calibration depends on: two backends that agree on trajectories but not on
    the features of the obs_data (a mean over a cycle, a max, a max-minus-min) would calibrate
    to different parameters.
    """
    record_property('backend', backend)
    _skip_unless_available(backend)
    tight = {'CVODE_myokit': {'rtol': 1e-10, 'atol': 1e-12, 'MaximumStep': 1e-4},
             'casadi_integrator': {'method': 'cvodes', 'reltol': 1e-10, 'abstol': 1e-12}}
    ref_engine = _engine(tmp_path, models, 'CVODE_myokit', '3compartment',
                         solver_info=tight['CVODE_myokit'])
    engine = _engine(tmp_path, models, backend, '3compartment',
                     solver_info=tight.get(backend, _solver_info(backend, 0.001)))
    x0 = _x0(ref_engine)
    with contextlib.redirect_stdout(io.StringIO()):
        ref_cost, ref_obs = ref_engine.get_cost_and_obs_from_params(x0)
        cost, obs = engine.get_cost_and_obs_from_params(x0)
    # One row per data item's operand trace. Compared by the features a calibration reads, not
    # point by point: a valve closing one output step apart is 1e-4 of a trace at one point.
    obs, ref_obs = (np.asarray(o, dtype=float).reshape(len(engine.obs_info['operands']), -1)
                    for o in (obs, ref_obs))
    labels = [str(n) for n in engine.obs_info['operands']]
    gate = TOLERANCES[backend]['features']
    error, label = worst(feature_error(obs, ref_obs), labels)
    assert error < gate, f'{label}: feature error {error:.2e} (gate {gate:.0e})'
    assert float(cost) == pytest.approx(float(ref_cost), rel=_COST_TOLERANCE.get(backend, 1e-3))


# --------------------------------------------------------------------------- gradients


def _gradients(tmp_path, models):
    """(x0, Myokit FSA, CasADi AD, central FD) cost gradients for Lotka_Volterra.

    The FD is of the numeric cost on CVODE_myokit at rtol 1e-12, with a relative step of 1e-4:
    steps from 1e-4 to 1e-6 agree to ~1e-5 here. numdifftools' adaptive step is not used, as
    the cost is a max over the trace and its Richardson extrapolation lands on the wrong side
    of a grid switch (+0.36 for d/dgamma, against -0.115).
    3compartment is not used: CasADi's cvodes forward sensitivities fail on its valves.
    """
    fsa = _engine(tmp_path, models, 'CVODE_myokit', 'Lotka_Volterra', do_ad=True,
                  solver_info={'rtol': 1e-8, 'atol': 1e-10})
    ad = _engine(tmp_path, models, 'casadi_integrator', 'Lotka_Volterra', do_ad=True,
                 solver_info={'method': 'cvodes', 'reltol': 1e-10, 'abstol': 1e-12})
    fd_engine = _engine(tmp_path, models, 'CVODE_myokit', 'Lotka_Volterra',
                        solver_info={'rtol': 1e-12, 'atol': 1e-14, 'MaximumStep': 1e-3})
    x0 = _x0(fd_engine)
    with contextlib.redirect_stdout(io.StringIO()):
        g_fsa = np.ravel(np.asarray(fsa.get_gradient(x0), dtype=float))
        g_ad = np.ravel(np.asarray(ad.get_gradient(x0), dtype=float))
        g_fd = np.array([(float(fd_engine.get_cost_from_params(x0 + e)) -
                          float(fd_engine.get_cost_from_params(x0 - e))) / (2 * e[i])
                         for i, e in enumerate(np.diag(1e-4 * np.abs(x0)))])
    return x0, g_fsa, g_ad, g_fd


def _assert_gradient(label, g, g_fd):
    """Per component: within 1e-3 of the FD, relative to that component or to 1e-3 of the
    largest one, so a small component cannot hide behind a large one."""
    tolerance = 1e-3 * np.maximum(np.abs(g_fd), 1e-3 * np.max(np.abs(g_fd)))
    bad = np.abs(g - g_fd) > tolerance
    assert not bad.any(), f'{label} gradient {g} != FD {g_fd} in components {np.where(bad)[0]}'


def test_casadi_ad_gradient_matches_fd(models, tmp_path, record_property):
    record_property('backend', 'casadi_integrator')
    _skip_unless_available('casadi_integrator')
    _, _, g_ad, g_fd = _gradients(tmp_path, models)
    assert np.max(np.abs(g_fd)) > 0, 'a flat cost would make this vacuous'
    _assert_gradient('CasADi AD', g_ad, g_fd)


@pytest.mark.xfail(strict=True, reason=(
    'Myokit FSA cost gradient on Lotka_Volterra (max features) differs from the FD and from '
    'CasADi AD: d/dalpha 37.24 vs 37.86, d/dgamma -1.18 vs -0.115, on Myokit 1.39.1 and 1.40.0 '
    'alike, so not the sensitivity error control of #387.'))
def test_myokit_fsa_gradient_matches_fd(models, tmp_path, record_property):
    record_property('backend', 'CVODE_myokit')
    _, g_fsa, _, g_fd = _gradients(tmp_path, models)
    _assert_gradient('Myokit FSA', g_fsa, g_fd)


def test_casadi_ad_gradient_after_param_change_matches_fd(models, tmp_path, record_property):
    """A CasADi AD gradient taken after another one, at new parameters, is the gradient there.

    casadi_python caches its cvodes integrators process-wide, keyed by model, grid and options.
    A cache that also kept the first point's sensitivities would hand the second call a stale
    gradient; a fresh engine and a central difference at the second point both say what it
    should be.
    """
    record_property('backend', 'casadi_integrator')
    _skip_unless_available('casadi_integrator')
    info = {'method': 'cvodes', 'reltol': 1e-10, 'abstol': 1e-12}
    used = _engine(tmp_path, models, 'casadi_integrator', 'Lotka_Volterra', do_ad=True,
                   solver_info=info)
    fd_engine = _engine(tmp_path, models, 'CVODE_myokit', 'Lotka_Volterra',
                        solver_info={'rtol': 1e-12, 'atol': 1e-14, 'MaximumStep': 1e-3})
    x0 = _x0(fd_engine)
    x1 = x0 * np.array([1.2, 0.9, 1.1, 0.8])
    with contextlib.redirect_stdout(io.StringIO()):
        g0 = np.ravel(np.asarray(used.get_gradient(x0), dtype=float))
        g1 = np.ravel(np.asarray(used.get_gradient(x1), dtype=float))
        fresh = _engine(tmp_path / 'fresh', models, 'casadi_integrator', 'Lotka_Volterra',
                        do_ad=True, solver_info=info)
        g1_fresh = np.ravel(np.asarray(fresh.get_gradient(x1), dtype=float))
        fd = np.array([(fd_engine.get_cost_from_params(x1 + e) -
                        fd_engine.get_cost_from_params(x1 - e)) / (2 * e[i])
                       for i, e in enumerate(np.diag(1e-6 * np.abs(x1)))], dtype=float)
    assert np.max(np.abs(g1 - g0)) > 1e-3 * np.max(np.abs(g0)), 'the two points share a gradient'
    np.testing.assert_allclose(g1, g1_fresh, rtol=1e-8, atol=0)
    assert np.max(np.abs(g1 - fd)) < 1e-4 * np.max(np.abs(fd)), (g1, fd)


# --------------------------------------------------------------------------- model_type cpp


def _cpp_states(folder):
    """{state name without the _module suffix: values} from sol0D_states.txt."""
    path = os.path.join(folder, 'sol0D_states.txt')
    with open(path) as f:
        header = f.readline().lstrip('#').split(';')
    data = np.loadtxt(path, ndmin=2)
    columns = {}
    for i, item in enumerate(header):
        if ':' in item:
            columns[item.split(':', 1)[1].split('[')[0].strip()] = data[:, i]
    return data[:, 0], columns


def test_cpp_cvode_matches_the_reference(models, tmp_path, record_property):
    """model_type cpp with CVODE: generated, built and run, against the converged reference."""
    record_property('backend', 'cpp_CVODE')
    if shutil.which('cmake') is None:
        pytest.skip('cmake not found, which model_type cpp needs')
    import yaml
    from libcuflynx.scripts.script_generate_with_new_architecture import \
        generate_with_new_architecture
    model = '3compartment'
    sim_time, dt, _ = MODELS[model]
    with open(os.path.join(REPO, 'user_run_files', 'user_inputs.yaml')) as f:
        inp = yaml.safe_load(f)
    for key in ['user_inputs_path_override', 'resources_dir', 'generated_models_dir',
                'param_id_output_dir', 'param_id_obs_path']:
        inp.pop(key, None)
    inp.update(file_prefix=model, input_param_file=f'{model}_parameters.json', model_type='cpp',
               couple_to_1d=False, resources_dir=RESOURCES, pre_time=0.0, sim_time=sim_time,
               dt=dt, generated_models_dir=str(tmp_path / 'generated'),
               solver_info={'solver': 'CVODE', 'dt_solver': 1e-4,
                            'MaximumNumberOfSteps': 5000})
    with contextlib.redirect_stdout(io.StringIO()):
        assert generate_with_new_architecture(False, inp)
    cpp = tmp_path / 'generated' / model
    configure = ['cmake', '-S', str(cpp), '-B', str(cpp / 'build')]
    if os.environ.get('SUNDIALS_DIR'):
        configure.append(f"-DSUNDIALS_DIR={os.environ['SUNDIALS_DIR']}")
    for command in (configure, ['cmake', '--build', str(cpp / 'build'), '-j', '4']):
        done = subprocess.run(command, capture_output=True, text=True)
        assert done.returncode == 0, (done.stdout + done.stderr)[-3000:]
    run = subprocess.run([str(cpp / 'build' / 'main0d'), '-tEnd', str(sim_time), '-tSave', '0',
                          '-outDir', str(tmp_path / 'out')],
                         capture_output=True, text=True, timeout=300)
    assert run.returncode == 0, (run.stdout + run.stderr)[-3000:]

    names, ref = _reference(models, model)
    t, columns = _cpp_states(tmp_path / 'out')
    assert len(t) == ref.shape[1], (len(t), ref.shape)
    np.testing.assert_allclose(t, np.arange(ref.shape[1]) * dt, atol=1e-9)
    short = [n.split('/')[0].removesuffix('_module') + '/' + n.split('/')[1] for n in names]
    missing = [s for s in short if s not in columns]
    assert not missing, f'states not in the cpp output: {missing}'
    x = np.array([columns[s] for s in short])
    _assert_close(x, ref, names, TOLERANCES['cpp_CVODE'], f'cpp CVODE on {model}')


# --------------------------------------------------------------------------- division guard


_DIVISION_GUARD_INFO = {'CVODE_myokit': {'rtol': 1e-10, 'atol': 1e-12, 'MaximumStep': 1e-4,
                                         'MaximumNumberOfSteps': 10 ** 7}}


@pytest.fixture(scope='module')
def control_phys_asd(tmp_path_factory):
    """control_phys_asd as CellML and as a casadi_python module, with its state names.

    Generating it takes ~40 s, so it is a fixture: the test itself then takes ~25 s.
    """
    pytest.importorskip('casadi')
    from libcuflynx.generators.PythonGenerator import PythonGenerator
    folder = tmp_path_factory.mktemp('control_phys_asd')
    cellml = _Models(str(folder)).cellml('control_phys_asd')
    with contextlib.redirect_stdout(io.StringIO()):
        casadi_path = PythonGenerator(cellml, output_dir=str(folder / 'casadi'),
                                      module_name='control_phys_asd',
                                      casadi_compat=True).generate()
    return {'CVODE_myokit': cellml, 'casadi_integrator': casadi_path,
            'names': state_names(cellml)}


def test_division_guard_keeps_the_sign_on_a_generated_model(control_phys_asd,
                                                            record_property):
    """control_phys_asd's chemoreflex divides by quantities that change sign. casadi_python
    guards every division against zero; a guard that drops the sign sends the CO2 drive the
    wrong way. Checked where a calibration would see it: each state's mean, min and max over
    the run, against CVODE_myokit at rtol 1e-10. At rtol 1e-8 this model's control takes a
    different branch on SUNDIALS 6 than on 7 (chi_a peaks at 0.028 instead of 0.096), so the
    comparison run has to be converged."""
    record_property('backend', 'casadi_integrator')
    from libcuflynx.solver_wrappers import get_simulation_helper
    names = control_phys_asd['names']
    runs = {}
    for backend in ('CVODE_myokit', 'casadi_integrator'):
        with contextlib.redirect_stdout(io.StringIO()):
            helper = get_simulation_helper(
                model_path=control_phys_asd[backend], solver=backend,
                model_type=MODEL_TYPE[backend], dt=0.001, sim_time=1.0, pre_time=0.0,
                solver_info=_DIVISION_GUARD_INFO.get(backend, _solver_info(backend, 0.001)))
        runs[backend] = run_states(helper, names)
        _close(helper)
    error, state = worst(feature_error(runs['casadi_integrator'], runs['CVODE_myokit']), names)
    assert error < TOLERANCES['casadi_integrator']['features'] * 10, (error, state)


# --------------------------------------------------------------------------- coverage of the schema


@pytest.mark.unit
def test_every_solver_in_schema_has_a_matrix_row():
    """A new solver or method cannot land without a row here or a named test elsewhere."""
    from libcuflynx.parsers.PrimitiveParsers import SOLVER_SCHEMA
    uncovered = []
    for solvers in SOLVER_SCHEMA['solvers_by_model_type'].values():
        for solver in solvers:
            if solver in MATRIX:
                matrix_method = (_solver_info(solver, 0.001).get('method'))
                for method in SOLVER_SCHEMA['methods_by_solver'][solver]:
                    if method in (matrix_method, 'CVODE') or (solver, method) in COVERED_METHODS:
                        continue
                    uncovered.append(f'{solver}/{method}')
            elif solver not in COVERED_ELSEWHERE:
                uncovered.append(solver)
    assert not uncovered, (f'{uncovered}: add a MATRIX row in tests/test_solver_matrix.py, or '
                           'name the test that covers it in COVERED_ELSEWHERE/COVERED_METHODS')
    for backend, model in EXCLUDED:
        assert model not in MATRIX.get(backend, []), (backend, model)


if __name__ == '__main__':
    import sys
    if sys.argv[1:] == ['--write-golden']:
        import tempfile
        folder = tempfile.mkdtemp()
        names, states = reference_solution(_Models(folder).cellml('Lotka_Volterra'),
                                           *MODELS['Lotka_Volterra'])
        os.makedirs(os.path.dirname(GOLDEN), exist_ok=True)
        np.savez(GOLDEN, names=np.array(names), states=states)
        print(f'wrote {GOLDEN}')
