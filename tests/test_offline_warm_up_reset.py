"""An offline warm-up starts from the model's initial values every time, and can be cleared.

``run_offline_pre_and_set_default_state`` is called again whenever a protocol is set again (a
notebook that adds a data item sets its obs_data twice). The second warm-up must give the state
the first did, not continue from it, on every backend. ``clear_offline_default_state`` returns
the helper to the model's initial values: a protocol without ``offline_pre_time`` set after one
with it.

The libOpenCOR helper has the same two tests in test_libopencor_solver.py.
"""
import os

import numpy as np
import pytest

from libcuflynx.generators.PythonGenerator import PythonGenerator
from libcuflynx.solver_wrappers import get_simulation_helper

NAMES = ['Lotka_Volterra_module/x', 'Lotka_Volterra_module/y']
BACKENDS = ['cellml', 'python', 'casadi_python', 'aadc_python']


SOLVERS = {
    'cellml': ('CVODE_myokit', {}),
    'python': ('solve_ivp', {'method': 'RK45', 'rtol': 1e-8, 'atol': 1e-8, 'max_step': 0.005}),
    'casadi_python': ('casadi_integrator', {'method': 'cvodes', 'max_step_size': 0.005,
                                            'max_num_steps': 50000}),
    'aadc_python': ('aadc_semi_implicit', {}),
}
GENERATOR_FLAGS = {'python': {}, 'casadi_python': {'casadi_compat': True},
                   'aadc_python': {'aadc_compat': True}}


def _helper(backend, cellml, out_dir):
    solver, solver_info = SOLVERS[backend]
    model_path = cellml
    if backend != 'cellml':
        if backend != 'python':
            pytest.importorskip({'casadi_python': 'casadi', 'aadc_python': 'aadc'}[backend])
        model_dir = os.path.join(out_dir, backend)
        os.makedirs(model_dir, exist_ok=True)
        model_path = PythonGenerator(cellml, output_dir=model_dir,
                                     module_name=f'Lotka_Volterra_{backend}',
                                     **GENERATOR_FLAGS[backend]).generate()
    return get_simulation_helper(model_path=model_path, model_type=backend, solver=solver,
                                 dt=0.01, sim_time=0.5, pre_time=0.0,
                                 solver_info=dict(solver_info))


def _run(helper):
    helper.reset_states()
    helper.update_times(0.01, 0.0, 0.5, 0.0)  # as the protocol executor does before each run
    assert helper.run()
    return np.array([np.asarray(r, dtype=float).ravel()
                     for r in helper.get_results(NAMES, flatten=True)])


@pytest.fixture
def helper_factory(generated_cellml_model_factory, tmp_path):
    cellml = generated_cellml_model_factory('Lotka_Volterra', 'Lotka_Volterra_parameters.json')
    return lambda backend: _helper(backend, cellml, str(tmp_path))


@pytest.mark.integration
@pytest.mark.solver
@pytest.mark.parametrize('backend', BACKENDS)
def test_a_second_warm_up_starts_from_the_initial_values(helper_factory, backend):
    helper = helper_factory(backend)
    cold = _run(helper)
    helper.run_offline_pre_and_set_default_state(1.0)
    once = _run(helper)
    assert np.max(np.abs(once - cold)) > 1e-6, 'the warm-up did not move the start state'
    helper.run_offline_pre_and_set_default_state(1.0)
    twice = _run(helper)
    np.testing.assert_array_equal(twice, once)


@pytest.mark.integration
@pytest.mark.solver
@pytest.mark.parametrize('backend', BACKENDS)
def test_clearing_the_warm_up_returns_to_the_initial_values(helper_factory, backend):
    helper = helper_factory(backend)
    cold = _run(helper)
    helper.run_offline_pre_and_set_default_state(1.0)
    assert np.max(np.abs(_run(helper) - cold)) > 1e-6
    helper.clear_offline_default_state()
    np.testing.assert_array_equal(_run(helper), cold)
    helper.clear_offline_default_state()  # nothing to clear: no change
    np.testing.assert_array_equal(_run(helper), cold)
