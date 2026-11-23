'''The figure input kinds read from uq, combine and identifiability (profile likelihood) steps:
each loader on a result as the runner records it, and the check of a figure step's inputs.'''
import json
import os
from types import SimpleNamespace

import numpy as np
import pytest

from libcuflynx.calibration_workflow import figures
from libcuflynx.calibration_workflow.distributions import (DISTRIBUTION_DIR, StoredDistribution,
                                                           parameters_from_info)
from libcuflynx.calibration_workflow.spec import WorkflowError

pytestmark = pytest.mark.unit


def _check(kind, wanted, name='x'):
    '''check_figure_step on a figure step reading ``kind`` from the step ``wanted``.'''
    workflow = SimpleNamespace(step=lambda step_id: wanted)
    step = SimpleNamespace(id='fig', figure={'inputs': {name: {'kind': kind, 'step': wanted.id}}})
    figures.check_figure_step(workflow, step)


def _step(kind, **keys):
    return SimpleNamespace(**{'id': 'src', 'kind': kind, 'store_distribution': False,
                              'obs_data': 'obs.json', **keys})


@pytest.mark.parametrize('kind, step_kind', [
    ('values', 'combine'), ('distribution', 'uq'), ('laplace', 'uq'),
    ('params_for_id', 'uq'), ('obs_data', 'uq'), ('profile_likelihood', 'identifiability')])
def test_the_devel_step_kinds_are_read(kind, step_kind):
    _check(kind, _step(step_kind))


def test_a_calibration_distribution_still_needs_store_distribution():
    with pytest.raises(WorkflowError, match='store_distribution'):
        _check('distribution', _step('calibration'))
    _check('distribution', _step('calibration', store_distribution=True))


def test_a_profile_likelihood_is_not_read_from_a_uq_step():
    with pytest.raises(WorkflowError, match='identifiability step'):
        _check('profile_likelihood', _step('uq'))


def test_combined_values_are_named_as_the_target_names_them():
    result = {'calibrated': [{'vessel': 'heart', 'param': 'E_max', 'value': 2.5,
                              'combined_from': ['a', 'b']}]}
    assert figures.INPUT_KINDS['values'].load('run', 'mix', result) == {'heart/E_max': 2.5}


def test_a_uq_steps_distribution_is_loaded(tmp_path):
    parameters = parameters_from_info([['heart/E_max'], ['aorta/C']], [1.0, 0.1], [3.0, 1.0])
    samples = np.array([[2.0, 0.5], [2.5, 0.4], [1.5, 0.6]])
    StoredDistribution(parameters, samples, source={'step': 'post'}).save(
        os.path.join(tmp_path, 'post', DISTRIBUTION_DIR))
    loaded = figures.INPUT_KINDS['distribution'].load(str(tmp_path), 'post', {})
    assert loaded['names'] == ['E_max_heart', 'C_aorta']     # the model's names
    np.testing.assert_array_equal(loaded['samples'], samples)


def test_a_profile_likelihood_is_loaded_with_its_sweeps(tmp_path):
    path = os.path.join(tmp_path, 'param_0_E_max.json')
    with open(path, 'w') as f:
        json.dump({'param_name': 'heart/E_max', 'param_values': [1.0, 2.0, 3.0],
                   'costs': [4.0, 1.0, 5.0], 'best_val': 2.0, 'best_cost': 1.0,
                   'ci_95': [1.5, 2.6]}, f)
    result = {'profile_likelihood': {
        'dir': str(tmp_path), 'nll_threshold_95': 1.92,
        'profiles': [{'parameter': 'heart/E_max', 'file': path, 'best_val': 2.0,
                      'best_cost': 1.0, 'ci_95': [1.5, 2.6]}]}}
    loaded = figures.INPUT_KINDS['profile_likelihood'].load(str(tmp_path), 'ia', result)
    assert loaded['nll_threshold_95'] == 1.92
    (profile,) = loaded['profiles']
    assert profile['parameter'] == 'heart/E_max' and profile['ci_95'] == [1.5, 2.6]
    np.testing.assert_array_equal(profile['param_values'], [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(profile['costs'], [4.0, 1.0, 5.0])


def test_a_step_with_no_profile_likelihood_says_so():
    with pytest.raises(WorkflowError, match='no profile likelihood'):
        figures.INPUT_KINDS['profile_likelihood'].load('run', 'ia', {})
