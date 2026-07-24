"""Unit tests for the Laplace approximation's gradient-source Hessian (Fisher information) and
its condition-number guard (issue #293 follow-up). These exercise the numerics with a mock
param-id engine, so no model/calibration is needed."""
import os

import numpy as np
import pytest

from identifiabilty_analysis.identifiabilityAnalysis import IdentifiabilityAnalysis
from parsers.PrimitiveParsers import scriptFunctionParser
from utilities.utility_funcs import extract_hessian_from_samples, _param_fit_scale


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


def test_laplace_succeeds_on_wide_magnitude_but_identifiable_params(tmp_path):
    # J columns spanning ~16 orders of magnitude (as 3compartment's parameters do) but with
    # *independent* (diagonal) sensitivities: the parameters are perfectly identifiable, only
    # differently scaled. The raw FIM = diag(1e16, 1e-16) has condition number 1e32, so the old
    # code aborted; Jacobi normalisation removes the scaling and it now succeeds with the correct
    # covariance = inv(FIM) = diag(1e-16, 1e16). This is the core of the #293 fix.
    sens = {'obsA': {'p0': 1e8, 'p1': 0.0}, 'obsB': {'p0': 0.0, 'p1': 1e-8}}
    pid = _MockParamId(sens, ['p0', 'p1'], ['obsA', 'obsB'], [1.0, 1.0])
    ia = _make_ia(pid, [1.0, 1.0], str(tmp_path))
    ia.run_laplace_approximation({'method': 'Laplace', 'gradient_source': 'AD'})
    fim = np.array([[1e16, 0.0], [0.0, 1e-16]])
    cov = ia.covariance_matrix_Laplace
    assert np.allclose(cov, np.linalg.inv(fim), rtol=1e-9) and np.all(np.isfinite(cov))
    # p0 is tightly constrained (tiny variance), p1 barely constrained (huge variance) -- a real,
    # correct answer, not the massively-inflated garbage the un-normalised inverse produced.
    assert cov[0, 0] == pytest.approx(1e-16, rel=1e-9)
    assert cov[1, 1] == pytest.approx(1e16, rel=1e-9)
    saved = np.load(os.path.join(str(tmp_path), 'mock_laplace_covariance.npy'))
    assert np.allclose(saved, cov)


def test_laplace_raises_on_genuinely_correlated_params(tmp_path):
    # Two observables whose sensitivity rows are parallel (obsB = 2 * obsA), so the parameters
    # only ever appear in a fixed linear combination: genuinely (jointly) non-identifiable. The
    # FIM stays singular *after* normalisation (unit-diagonal, off-diagonal == 1), so this must
    # still raise -- normalisation fixes scaling, not real correlation.
    sens = {'obsA': {'p0': 1.0, 'p1': 2.0}, 'obsB': {'p0': 2.0, 'p1': 4.0}}
    pid = _MockParamId(sens, ['p0', 'p1'], ['obsA', 'obsB'], [1.0, 1.0])
    ia = _make_ia(pid, [1.0, 1.0], str(tmp_path))
    with pytest.raises(RuntimeError, match="ill-conditioned"):
        ia.run_laplace_approximation({'method': 'Laplace', 'gradient_source': 'AD'})
    assert not os.path.exists(os.path.join(str(tmp_path), 'mock_laplace_covariance.npy'))


def test_laplace_raises_on_information_less_parameter(tmp_path):
    # p1 has zero sensitivity in every observable, so the FIM diagonal for p1 is 0: that parameter
    # carries no information and its variance is undefined (Jacobi scaling would divide by zero).
    # Must raise a clear non-identifiability error rather than produce nan/inf.
    sens = {'obsA': {'p0': 2.0, 'p1': 0.0}, 'obsB': {'p0': 1.0, 'p1': 0.0}}
    pid = _MockParamId(sens, ['p0', 'p1'], ['obsA', 'obsB'], [1.0, 1.0])
    ia = _make_ia(pid, [1.0, 1.0], str(tmp_path))
    with pytest.raises(RuntimeError, match="non-positive diagonal|no.*curvature"):
        ia.run_laplace_approximation({'method': 'Laplace', 'gradient_source': 'AD'})
    assert not os.path.exists(os.path.join(str(tmp_path), 'mock_laplace_covariance.npy'))


def test_laplace_invert_tolerates_mildly_indefinite_hessian(tmp_path, capsys):
    # A slightly negative curvature (FD / optimiser noise at a rough optimum, e.g. the test_fft and
    # 3compartment debug calibrations) must still give a finite covariance -- the plain inverse
    # always did -- rather than aborting the whole identifiability run. C == inv(P) exactly; the
    # non-positive variance is flagged, not fatal.
    ia = _make_ia(_well_conditioned_pid(), [1.0, 1.0], str(tmp_path))
    precision = np.array([[-2.0, 0.1], [0.1, 3.0]])  # indefinite but well-conditioned
    cov, cond = ia._invert_precision_normalised(precision, 'FD')
    assert np.allclose(cov, np.linalg.inv(precision))
    assert np.all(np.isfinite(cov)) and np.isfinite(cond)
    assert cov[0, 0] < 0  # negative variance for the negative-curvature parameter
    assert 'non-positive variance' in capsys.readouterr().out


def test_laplace_invert_raises_on_zero_curvature(tmp_path):
    # Exactly-zero curvature is genuinely information-less (infinite variance): still a hard error.
    ia = _make_ia(_well_conditioned_pid(), [1.0, 1.0], str(tmp_path))
    precision = np.array([[2.0, 0.0], [0.0, 0.0]])
    with pytest.raises(RuntimeError, match="no.*curvature"):
        ia._invert_precision_normalised(precision, 'FD')


def _quadratic_samples_and_losses(center, scale, H_true, half_frac=0.02, n=40, seed=0):
    """Latin-hypercube-ish samples of an *exact* quadratic loss 0.5 (p-c)^T H_true (p-c), with each
    parameter perturbed by ``half_frac`` of its scale. Returns (samples, losses)."""
    center = np.asarray(center, float)
    scale = np.asarray(scale, float)
    rng = np.random.default_rng(seed)
    z = (rng.random((n, center.size)) - 0.5) * 2 * half_frac  # normalised perturbations
    samples = center + z * scale
    d = samples - center
    losses = 0.5 * np.einsum('ni,ij,nj->n', d, H_true, d)
    return samples, losses


def test_hessian_fit_normalised_recovers_wide_magnitude_curvature():
    # Parameters spanning 16 orders of magnitude (as 3compartment's do). The true Hessian is O(1)
    # in scale-normalised coordinates but spans ~1e32 in raw space. Fitting in normalised
    # coordinates recovers it exactly; the raw-space fit cannot (its design matrix is singular to
    # working precision), which is what produced the indefinite, non-positive-diagonal Hessian.
    center = np.array([1e-8, 1e8])
    scale = np.array([1e-8, 1e8])
    H_norm = np.array([[2.0, 0.5], [0.5, 3.0]])
    H_true = H_norm / np.outer(scale, scale)  # [[2e16, 0.5], [0.5, 3e-16]]
    samples, losses = _quadratic_samples_and_losses(center, scale, H_true)

    H_fit = extract_hessian_from_samples(samples, losses, scale=scale, center=center)
    assert np.allclose(H_fit, H_true, rtol=1e-6), H_fit
    # Diagonal is correctly positive (identifiable), unlike the raw-space fit.
    assert H_fit[0, 0] > 0 and H_fit[1, 1] > 0

    H_raw = extract_hessian_from_samples(samples, losses)  # no normalisation
    # The large-magnitude curvature (2e16) is not recoverable in raw space.
    assert not np.isclose(H_raw[0, 0], H_true[0, 0], rtol=1e-2)


def test_param_fit_scale_uses_ranges_then_falls_back():
    class _PidWithBounds:
        param_id_info = {'param_mins': np.array([1e8, 1e-9]),
                         'param_maxs': np.array([5e8, 5e-8])}

    scale = _param_fit_scale(_PidWithBounds(), np.array([1.5e8, 1e-8]))
    assert np.allclose(scale, [4e8, 4.9e-8])

    # No bounds -> fall back to |best|; zeros -> 1 (never divide by zero).
    class _PidNoBounds:
        param_id_info = {}

    scale2 = _param_fit_scale(_PidNoBounds(), np.array([3.0, 0.0]))
    assert np.allclose(scale2, [3.0, 1.0])


def test_laplace_gradient_source_not_available_for_model(tmp_path):
    # AD is not a valid source for cellml_only + CVODE_opencor (only FD is). Must raise clearly.
    pid = _well_conditioned_pid()
    pid.model_type = 'cellml_only'
    pid.solver_info = {'solver': 'CVODE_opencor'}
    ia = _make_ia(pid, [1.0, 1.0], str(tmp_path))
    with pytest.raises(ValueError, match="not available"):
        ia.run_laplace_approximation({'method': 'Laplace', 'gradient_source': 'AD'})
