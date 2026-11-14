"""Smoke test of the 3compartment parameter-reduction study (benchmarks/param_reduction_3compartment):
synthetic obs_data, the workflow (Sobol screen, PCA and top-N reductions, one CMA-ES fit of the PCA set)
and compare.py, each as its own process at a tiny budget."""

import json
import os
import subprocess
import sys
import time

import pytest

STUDY = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     'benchmarks', 'param_reduction_3compartment')


#: wall-clock budget for the three processes together, under the 10-minute limit on a test
TOTAL_SECONDS = 240


def _run(deadline, script, *args):
    """Run ``script``; it gets what is left of the shared ``deadline`` (time.monotonic)."""
    proc = subprocess.run([sys.executable, os.path.join(STUDY, script), *map(str, args)],
                          cwd=STUDY, capture_output=True, text=True,
                          timeout=max(1.0, deadline - time.monotonic()))
    assert proc.returncode == 0, f'{script} failed:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}'
    return proc.stdout


@pytest.mark.integration
def test_param_reduction_study_smoke(tmp_path):
    out = str(tmp_path / 'study')
    deadline = time.monotonic() + TOTAL_SECONDS
    _run(deadline, 'make_synthetic_obs.py', out)
    assert os.path.isfile(os.path.join(out, 'inputs', 'truth.json'))

    _run(deadline, 'run_study.py', out, '--num-samples', 2, '--calls', 2, '--seeds', 0,
         '--methods', 'pca')
    run_dir = os.path.join(out, 'run')
    for step in ('screen', 'reduce_pca', 'reduce_simple',
                 'fit_pca_s0'):
        assert os.path.isfile(os.path.join(run_dir, step, 'step_result.json')), step
    with open(os.path.join(run_dir, 'reduce_pca', 'param_reduction.json')) as f:
        report = json.load(f)
    # the result page's text reads the options and the mode count from this report
    sys.path.insert(0, STUDY)
    try:
        from page_text import pca_options_text
    finally:
        sys.path.remove(STUDY)
    text = pca_options_text(report)
    assert f'min mode std {report["reduction"]["min_mode_std"]}' in text
    assert f'Found {report["n_informative_modes"]} informative modes' in text

    _run(deadline, 'compare.py', out)
    with open(os.path.join(out, 'compare', 'summary.json')) as f:
        summary = json.load(f)
    assert {fit['method'] for fit in summary['fits']} == {'pca'}
    assert len(summary['parameters']) == 7
    assert all(fit['chi2'] >= 0 and fit['calibrated'] for fit in summary['fits'])
    assert set(summary['reductions']) == {'pca', 'simple'}
    assert os.path.isfile(os.path.join(out, 'compare', 'traces.json'))
