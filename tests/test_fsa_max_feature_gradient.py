"""The Myokit FSA gradient of a max feature whose trace has two near-equal peaks.

Lotka_Volterra's y peaks at 51.8036 (t=0.59) and 51.8055 (t=4.08). The max feature is y at the
second peak, so its derivative is d y/d p there. A central difference along the CVODES
sensitivity S with a step large enough to move the max to the first peak on one side averages in
that peak's sensitivity instead.

The built-in max has an exact rule (S at the argmax). The reference for it is the chain rule over
the same run's own sensitivities, read at each feature's argmax: it shares every number with the
gradient except how the features are differentiated. A user operation has no rule, so it takes
the difference along S, checked against a finite difference of the cost.
"""
import contextlib
import io
import json
import os

import numpy as np
import pytest

pytestmark = pytest.mark.integration

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESOURCES = os.path.join(REPO, 'resources')


@pytest.fixture(scope='module')
def lotka_volterra_cellml(tmp_path_factory):
    from libcuflynx.scripts.script_generate_with_new_architecture import \
        generate_with_new_architecture
    from libcuflynx.utilities.utility_funcs import get_default_inp_data_dict
    folder = str(tmp_path_factory.mktemp('lv_models'))
    inp = get_default_inp_data_dict('Lotka_Volterra', 'Lotka_Volterra_parameters.json', RESOURCES)
    inp.update(generated_models_dir=folder, DEBUG=True, model_type='cellml')
    with contextlib.redirect_stdout(io.StringIO()):
        assert generate_with_new_architecture(False, inp)
    return os.path.join(folder, 'Lotka_Volterra', 'Lotka_Volterra.cellml')


_CUSTOM_MAX = """import numpy as np
from libcuflynx.param_id.operation_funcs import series_to_constant


@series_to_constant
def custom_max(x, series_output=False):
    if series_output:
        return x
    return float(np.max(x))
"""


def _engine(tmp_path, cellml, operation=None, do_ad=True, solver_info=None):
    """An engine on Lotka_Volterra's obs_data (max of x and of y), gaussian_MLE pinned.

    ``operation='custom_max'`` swaps both maxes for a user operation with no exact rule.
    """
    from libcuflynx.param_id.paramID import CVS0DParamID
    os.makedirs(str(tmp_path), exist_ok=True)
    with open(os.path.join(RESOURCES, 'Lotka_Volterra_obs_data.json')) as f:
        obs = json.load(f)
    for item in obs['data_items']:
        item['cost_type'] = 'gaussian_MLE'
        if operation:
            item['operation'] = operation
    ops_path = os.path.join(str(tmp_path), 'custom_ops.py')
    with open(ops_path, 'w') as f:
        f.write(_CUSTOM_MAX)
    obs_path = os.path.join(str(tmp_path), f'obs_{operation}_{do_ad}.json')
    with open(obs_path, 'w') as f:
        json.dump(obs, f)
    with contextlib.redirect_stdout(io.StringIO()):
        outer = CVS0DParamID(
            model_path=cellml, model_type='cellml', param_id_method='sp_minimize',
            file_name_prefix='Lotka_Volterra',
            params_for_id_path=os.path.join(RESOURCES, 'Lotka_Volterra_params_for_id.json'),
            param_id_obs_path=obs_path, sim_time=None, pre_time=None, dt=0.01,
            solver_info=solver_info or {'solver': 'CVODE_myokit', 'rtol': 1e-8, 'atol': 1e-10},
            do_ad=do_ad, DEBUG=True, param_id_output_dir=str(tmp_path),
            operation_funcs_external_path=ops_path)
    engine = outer.param_id
    engine.output_dir = str(tmp_path)
    return engine, obs['data_items']


def _x0(engine):
    raw = engine.sim_helper.get_init_param_vals(engine.param_id_info['param_names'])
    return np.array([np.ravel(v)[0] for v in raw], dtype=float)


def _features_and_sensitivities(engine, x0):
    """Each item's operand trace and its CVODES sensitivity [time, param], from one FSA run."""
    engine._ensure_fsa_setup()
    with contextlib.redirect_stdout(io.StringIO()):
        _, operands_list, _ = engine.get_cost_obs_and_pred_from_params(
            x0, reset=True, only_one_exp=0)
    sens = np.asarray(engine.sim_helper._fsa_sensitivities_history[0], dtype=float)
    deps = engine._fsa_dependent_names
    traces = [np.asarray(entry[0], dtype=float) for entry in operands_list[0]]
    columns = [sens[:, deps.index(ops[0]), :] for ops in engine.obs_info['operands']]
    return traces, columns


def test_y_has_two_near_equal_peaks(lotka_volterra_cellml, tmp_path):
    """The case this file is about: the max of y is within 1e-4 of another local max."""
    engine, _ = _engine(tmp_path, lotka_volterra_cellml)
    traces, _ = _features_and_sensitivities(engine, _x0(engine))
    y = traces[1]
    peaks = [i for i in range(1, len(y) - 1) if y[i - 1] < y[i] >= y[i + 1]]
    top = sorted(y[peaks])[-2:]
    assert len(peaks) >= 2 and (top[1] - top[0]) < 1e-4 * top[1]


def test_fsa_cost_gradient_is_the_chain_rule_at_the_argmax(lotka_volterra_cellml, tmp_path):
    engine, items = _engine(tmp_path, lotka_volterra_cellml)
    x0 = _x0(engine)
    with contextlib.redirect_stdout(io.StringIO()):
        grad = np.ravel(np.asarray(engine.get_gradient(x0), dtype=float))
    traces, columns = _features_and_sensitivities(engine, x0)
    # gaussian_MLE: 0.5 * sum_k w_k ((f_k - v_k)/std_k)^2 over the weighted denominator.
    by_hand = sum(item['weight'] * (trace.max() - item['value']) / item['std'] ** 2
                  * column[np.argmax(trace)]
                  for item, trace, column in zip(items, traces, columns))
    by_hand = by_hand / float(engine._total_weighted_obs_denominator())
    np.testing.assert_allclose(grad, by_hand, rtol=1e-6)


def test_fsa_feature_sensitivity_is_the_sensitivity_at_the_argmax(lotka_volterra_cellml,
                                                                   tmp_path):
    engine, _ = _engine(tmp_path, lotka_volterra_cellml)
    x0 = _x0(engine)
    with contextlib.redirect_stdout(io.StringIO()):
        sens = engine.get_observable_sensitivities(x0, gradient_method='FSA')
    traces, columns = _features_and_sensitivities(engine, x0)
    labels = [engine._observable_label(i) for i in engine.obs_info['const_idx_to_obs_idx']]
    for label, trace, column in zip(labels, traces, columns):
        got = np.array(list(sens[label].values()), dtype=float)
        np.testing.assert_allclose(got, column[np.argmax(trace)], rtol=1e-6)


def test_a_user_operation_takes_the_difference_along_s_and_matches_fd(lotka_volterra_cellml,
                                                                      tmp_path):
    from libcuflynx.param_id import fsa_backend
    engine, _ = _engine(tmp_path / 'fsa', lotka_volterra_cellml, operation='custom_max')
    x0 = _x0(engine)
    with contextlib.redirect_stdout(io.StringIO()):
        grad = np.ravel(np.asarray(engine.get_gradient(x0), dtype=float))
        _, operands_list, _ = engine.get_cost_obs_and_pred_from_params(
            x0, reset=True, only_one_exp=0)
    sens = fsa_backend.operand_sensitivities(
        engine.sim_helper, engine._fsa_dependent_names, engine._fsa_param_names_flat)
    entry = fsa_backend.combined_entry_sensitivities(engine, sens, 0)
    assert fsa_backend.exact_obs_tangent(engine, operands_list[0], entry, 0, 0) is None

    tight = {'solver': 'CVODE_myokit', 'rtol': 1e-12, 'atol': 1e-14, 'MaximumStep': 1e-3}
    fd_engine, _ = _engine(tmp_path / 'fd', lotka_volterra_cellml, operation='custom_max',
                           do_ad=False, solver_info=tight)
    with contextlib.redirect_stdout(io.StringIO()):
        fd = np.array([(float(fd_engine.get_cost_from_params(x0 + e))
                        - float(fd_engine.get_cost_from_params(x0 - e))) / (2 * e[i])
                       for i, e in enumerate(np.diag(1e-4 * np.abs(x0)))])
    # Per component, within 1e-3 of the FD or of 1e-3 of its largest component: the FSA run is
    # at rtol 1e-8, which leaves d/dgamma (-0.115, beside d/dbeta -882) ~5e-4 off.
    tolerance = 1e-3 * np.maximum(np.abs(fd), 1e-3 * np.max(np.abs(fd)))
    assert np.all(np.abs(grad - fd) <= tolerance), f'{grad} != FD {fd}'


@pytest.mark.unit
def test_every_exact_tangent_rule_is_the_derivative_of_its_operation():
    """Each built-in rule against a central difference of its own operation, on random traces
    (no ties, so every max/min is differentiable), with the operation's default kwargs."""
    from libcuflynx.param_id.operation_funcs import get_operation_funcs_dict_for_mode
    rng = np.random.default_rng(0)
    funcs = get_operation_funcs_dict_for_mode('numpy')
    ruled = {name: f for name, f in funcs.items() if getattr(f, 'exact_tangent', None)}
    assert {'max', 'min', 'mean', 'max_minus_min', 'addition', 'subtraction', 'multiplication',
            'division', 'mean_in_range', 'max_in_range', 'min_in_range',
            'max_minus_min_in_range'} <= set(ruled)
    for name, func in ruled.items():
        xs = [rng.uniform(1.0, 2.0, 50), rng.uniform(1.0, 2.0, 50)]
        ts = [rng.normal(size=50), rng.normal(size=50)]
        n = 2 if name in ('addition', 'subtraction', 'multiplication', 'division') else 1
        xs, ts = xs[:n], ts[:n]
        h = 1e-7
        fd = (np.asarray(func(*[x + h * t for x, t in zip(xs, ts)]))
              - np.asarray(func(*[x - h * t for x, t in zip(xs, ts)]))) / (2 * h)
        np.testing.assert_allclose(func.exact_tangent(xs, ts), fd, rtol=1e-6, atol=1e-8,
                                   err_msg=name)
