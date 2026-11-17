"""The local sensitivity analysis's signed relative coefficients, d ln Y / d ln p, and their row
labels, which are the Sobol indices' so one heatmap can show both."""
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from libcuflynx.param_id.paramID import observable_labels
from libcuflynx.sensitivity_analysis.sensitivityAnalysis import (LOCAL_SIGNED_FILE,
                                                                 SensitivityAnalysis)
from libcuflynx.sensitivity_analysis.sobolSA import observable_indices, sobol_SA

K, B, C = 2.0, 3.0, 6.0
OBS_INFO = {'num_obs': 3, 'item_names_for_plotting': ['dec', 'inc', 'zero'],
            'experiment_idxs': [0, 0, 1], 'subexperiment_idxs': [0, 0, 0],
            'operations': ['max', 'max', 'mean'], 'weight_const_vec': [1.0, 1.0, 1.0],
            'const_idx_to_obs_idx': [0, 1, 2]}


class _Engine:
    """A model with three scalar features of k and b: ``dec = C / k`` (decreasing in k),
    ``inc = -b k`` (negative, its magnitude increasing in k and b) and ``zero = 0``."""
    obs_info = OBS_INFO
    param_id_info = {'param_names': [['A/k'], ['A/b']], 'param_labels': ['A/k', 'A/b']}
    sim_helper = SimpleNamespace(get_init_param_vals=lambda names: [K, B])

    def _observable_label(self, i):
        return observable_labels(self.obs_info)[i]

    def get_cost_obs_and_pred_from_params(self, p, reset=True, only_one_exp=0):
        return None, [p], None

    def get_obs_output_dict(self, p):
        k, b = p
        return {'const': [C / k, -b * k, 0.0]}

    def get_observable_sensitivities(self, p, gradient_method=None, fd_rel_step=None):
        k, b = p
        rows = [{'A/k': -C / k ** 2, 'A/b': 0.0}, {'A/k': -b, 'A/b': -k},
                {'A/k': 0.0, 'A/b': 0.0}]
        return {self._observable_label(i): row for i, row in enumerate(rows)}


def _analysis(out_dir):
    sa = SensitivityAnalysis.__new__(SensitivityAnalysis)
    sa.sa_options = {'method': 'local'}
    sa.SA_manager = SimpleNamespace(include_prediction_items=False, output_dir=str(out_dir))
    sa._local_engine = SimpleNamespace(param_id=_Engine())
    sa._record_in_study = lambda method: None
    sa.local_sensitivities = None
    return sa


@pytest.mark.unit
def test_the_signed_coefficient_is_negative_for_a_decreasing_output(tmp_path):
    local = _analysis(tmp_path).run_local_sensitivity()
    signed = pd.read_csv(os.path.join(tmp_path, LOCAL_SIGNED_FILE), index_col='output')
    assert list(signed.columns) == ['A/k', 'A/b']
    # d ln(C/k) / d ln k = -1
    assert signed.loc['dec (Exp0, Sub0)', 'A/k'] == pytest.approx(-1.0)
    # d ln(-bk) / d ln k = +1: the unsigned file, |p| dY/dp / |Y|, says -1 for a negative Y
    assert signed.loc['inc (Exp0, Sub0)'].tolist() == pytest.approx([1.0, 1.0])
    assert local['relative'][1].tolist() == pytest.approx([-1.0, -1.0])
    assert signed.loc['dec (Exp0, Sub0)', 'A/b'] == 0.0
    # undefined where the feature is 0: NaN, where the unsigned file says 0.0
    assert signed.loc['zero (Exp1, Sub0)'].isna().all()
    assert np.all(local['relative'][2] == 0.0)
    assert local['nominal_feature_vals'] == pytest.approx([C / K, -B * K, 0.0])


@pytest.mark.unit
def test_the_signed_rows_have_the_sobol_outputs_labels(tmp_path):
    local = _analysis(tmp_path).run_local_sensitivity()
    sobol = SimpleNamespace(obs_info=OBS_INFO, prediction_info=None,
                            _observable_indices=lambda: observable_indices(OBS_INFO),
                            _prediction_labels=lambda: [], _prediction_indices=lambda: [])
    columns, _ = sobol_SA._output_columns(sobol, 3)
    assert local['signed_output_names'] == columns
    signed = pd.read_csv(os.path.join(tmp_path, LOCAL_SIGNED_FILE), index_col='output')
    assert list(signed.index) == columns


@pytest.mark.unit
def test_an_item_named_in_two_experiments_keeps_the_sobol_label(tmp_path):
    """One name in two experiments: the engine's label is ``y [exp 1, sub 0]``, the Sobol
    output's ``y (Exp1, Sub0)``; the signed row has the Sobol one."""
    obs_info = dict(OBS_INFO, item_names_for_plotting=['y', 'zero', 'y'])

    class Engine(_Engine):
        pass
    Engine.obs_info = obs_info
    sa = _analysis(tmp_path)
    sa._local_engine = SimpleNamespace(param_id=Engine())
    local = sa.run_local_sensitivity()
    sobol = SimpleNamespace(obs_info=obs_info, prediction_info=None,
                            _observable_indices=lambda: observable_indices(obs_info),
                            _prediction_labels=lambda: [], _prediction_indices=lambda: [])
    columns, _ = sobol_SA._output_columns(sobol, 3)
    assert local['output_names'][2] == 'y [exp 1, sub 0]'
    assert local['signed_output_names'] == columns == ['y (Exp0, Sub0)', 'zero (Exp0, Sub0)',
                                                       'y (Exp1, Sub0)']
