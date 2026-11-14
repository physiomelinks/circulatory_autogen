"""The Myokit FSA gradient of a max feature whose trace has two near-equal peaks.

Lotka_Volterra's y peaks at 51.8036 (t=0.59) and 51.8055 (t=4.08). The max feature is y at the
second peak, so its derivative is d y/d p there. The FSA gradient is a central difference along
the CVODES sensitivity S on the fixed operand traces; a step large enough to move the max to the
first peak on one side averages in that peak's sensitivity instead.

The reference is the chain rule over the same run's own sensitivities, read at each feature's
argmax: it shares every number with the gradient except the differencing, so it isolates it.
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


def _engine(tmp_path, cellml):
    """An FSA engine on Lotka_Volterra's obs_data (max of x and of y), gaussian_MLE pinned."""
    from libcuflynx.param_id.paramID import CVS0DParamID
    with open(os.path.join(RESOURCES, 'Lotka_Volterra_obs_data.json')) as f:
        obs = json.load(f)
    for item in obs['data_items']:
        item['cost_type'] = 'gaussian_MLE'
    obs_path = os.path.join(str(tmp_path), 'obs.json')
    with open(obs_path, 'w') as f:
        json.dump(obs, f)
    with contextlib.redirect_stdout(io.StringIO()):
        outer = CVS0DParamID(
            model_path=cellml, model_type='cellml', param_id_method='sp_minimize',
            file_name_prefix='Lotka_Volterra',
            params_for_id_path=os.path.join(RESOURCES, 'Lotka_Volterra_params_for_id.json'),
            param_id_obs_path=obs_path, sim_time=None, pre_time=None, dt=0.01,
            solver_info={'solver': 'CVODE_myokit', 'rtol': 1e-8, 'atol': 1e-10},
            do_ad=True, DEBUG=True, param_id_output_dir=str(tmp_path))
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
