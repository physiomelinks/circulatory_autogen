"""Unit tests for the Laplace approximation's gradient-source Hessian (Fisher information) and
its condition-number guard (issue #293 follow-up). These exercise the numerics with a mock
param-id engine, so no model/calibration is needed."""
import os

import numpy as np
import pytest

from identifiabilty_analysis.identifiabilityAnalysis import IdentifiabilityAnalysis
from parsers.PrimitiveParsers import scriptFunctionParser


class _MockParamId:
    """Minimal stand-in exposing exactly what _fisher_information_matrix / run_laplace read."""

    def __init__(self, sens, param_names, labels, stds, model_type='casadi_python',
                 solver='casadi_integrator'):
        self.model_type = model_type
        self.solver_info = {'solver': solver}
        self._sens = sens                 # {obs_label: {param_name: d(feature)/d(param)}}
        self._labels = labels             # obs_idx -> label
        self.param_id_info = {'param_names': [[p] for p in param_names]}
        self.obs_info = {
            'const_idx_to_obs_idx': list(range(len(stds))),
            'std_const_vec': np.asarray(stds, dtype=float),
        }
        self.cost_type = 'gaussian_MLE'
        self.cost_funcs_dict = scriptFunctionParser().get_cost_funcs_dict('numpy')

    def _observable_label(self, obs_idx):
        return self._labels[obs_idx]

    def get_observable_sensitivities(self, param_vals):
        return self._sens


def _make_ia(param_id, best, tmp):
    ia = object.__new__(IdentifiabilityAnalysis)
    ia.param_id = param_id
    ia.best_param_vals = np.asarray(best, dtype=float)
    ia.file_name_prefix = 'mock'
    ia.param_id_output_dir = os.path.join(tmp, 'param_id_output')  # save dir is its parent (tmp)
    ia.rank = 0
    ia.covariance_matrix_Laplace = None
    ia.mean_Laplace = None
    return ia


def _well_conditioned_pid():
    # J = [[2,1],[0,3]], std = [1,2] -> FIM = [[4,2],[2,3.25]]
    sens = {'obsA': {'p0': 2.0, 'p1': 1.0}, 'obsB': {'p0': 0.0, 'p1': 3.0}}
    return _MockParamId(sens, ['p0', 'p1'], ['obsA', 'obsB'], [1.0, 2.0])


def test_fisher_information_matrix_matches_J_T_W_J(tmp_path):
    ia = _make_ia(_well_conditioned_pid(), [1.0, 1.0], str(tmp_path))
    fim = ia._fisher_information_matrix('AD')
    expected = np.array([[4.0, 2.0], [2.0, 3.25]])
    assert np.allclose(fim, expected), fim


def test_laplace_writes_inverse_fim_when_well_conditioned(tmp_path):
    ia = _make_ia(_well_conditioned_pid(), [1.0, 1.0], str(tmp_path))
    ia.run_laplace_approximation({'method': 'Laplace', 'gradient_source': 'AD'})
    fim = np.array([[4.0, 2.0], [2.0, 3.25]])
    assert np.allclose(ia.covariance_matrix_Laplace, np.linalg.inv(fim))
    cov = np.load(os.path.join(str(tmp_path), 'mock_laplace_covariance.npy'))
    assert np.allclose(cov, np.linalg.inv(fim)) and np.all(np.isfinite(cov))


def test_laplace_raises_on_ill_conditioned_fim(tmp_path):
    # J columns spanning ~16 orders of magnitude (as 3compartment's parameters do) -> the FIM is
    # ill-conditioned, so inverting it would give meaningless uncertainties. Must raise, not save.
    sens = {'obsA': {'p0': 1e8, 'p1': 0.0}, 'obsB': {'p0': 0.0, 'p1': 1e-8}}
    pid = _MockParamId(sens, ['p0', 'p1'], ['obsA', 'obsB'], [1.0, 1.0])
    ia = _make_ia(pid, [1.0, 1.0], str(tmp_path))
    with pytest.raises(RuntimeError, match="ill-conditioned"):
        ia.run_laplace_approximation({'method': 'Laplace', 'gradient_source': 'AD'})
    assert not os.path.exists(os.path.join(str(tmp_path), 'mock_laplace_covariance.npy'))


def test_laplace_gradient_source_not_available_for_model(tmp_path):
    # AD is not a valid source for cellml_only + CVODE_opencor (only FD is). Must raise clearly.
    pid = _well_conditioned_pid()
    pid.model_type = 'cellml_only'
    pid.solver_info = {'solver': 'CVODE_opencor'}
    ia = _make_ia(pid, [1.0, 1.0], str(tmp_path))
    with pytest.raises(ValueError, match="not available"):
        ia.run_laplace_approximation({'method': 'Laplace', 'gradient_source': 'AD'})
