"""benchmarks/solver_comparison.py runs on one tiny model with whichever backend is installed."""
import importlib.util
import json
import re
import sys

import pytest

from benchmarks import solver_comparison

_QUICK = ['--models', 'lotka_volterra', '--sim-time', '1', '--repeats', '3']


def _run(tmp_path, backends):
    out = tmp_path / 'out'
    assert solver_comparison.main(_QUICK + ['--backends', backends, '--out', str(out),
                                            '--work-dir', str(tmp_path / 'work')]) == 0
    report = json.loads((out / 'solver_comparison.json').read_text())
    table = (out / 'solver_comparison.md').read_text()
    return report, table


def _check_row(row):
    assert 'error' not in row, row
    assert row['setup_s'] > 0 and row['first_run_s'] > 0 and row['run_median_s'] > 0
    assert row['repeats'] == 3


@pytest.mark.unit
def test_myokit_alone(tmp_path):
    report, table = _run(tmp_path, 'myokit')
    [model] = report['models']
    assert model['model'] == 'lotka_volterra' and model['states'] == 2
    assert model['reference'] == 'myokit'
    row = model['backends']['myokit']
    _check_row(row)
    assert row['max_rel_err'] == 0.0
    assert row['steps'] > 0 and row['rhs_evals'] > 0
    assert '| lotka_volterra | 2 | CVODE_myokit (reference) |' in table
    # the SUNDIALS build Myokit loaded is recorded, with the optimisation caveat
    assert set(row['sundials']) == {'path', 'version'}
    assert report['myokit_sundials'] == row['sundials']
    if sys.platform.startswith('linux'):
        assert 'libsundials' in row['sundials']['path']
        assert re.fullmatch(r'\d+\.\d+\.\d+', row['sundials']['version'])
    assert 'depend on SUNDIALS being an optimised build' in table


@pytest.mark.unit
@pytest.mark.need_libopencor
def test_libopencor_alone_and_against_myokit(tmp_path):
    report, _ = _run(tmp_path, 'libopencor')
    assert report['models'][0]['reference'] == 'libopencor'
    _check_row(report['models'][0]['backends']['libopencor'])

    report, _ = _run(tmp_path, 'myokit,libopencor')
    row = report['models'][0]['backends']['libopencor']
    _check_row(row)
    # Same CVODE, same settings and the same output grid.
    assert row['max_rel_err'] < 1e-6


@pytest.mark.unit
def test_a_missing_backend_is_skipped_with_its_reason(monkeypatch, tmp_path):
    real = importlib.util.find_spec
    monkeypatch.setattr(solver_comparison.importlib.util, 'find_spec',
                        lambda name, *a: None if name == 'casadi' else real(name, *a))
    report, table = _run(tmp_path, 'myokit,casadi')
    assert report['skipped_backends'] == {'casadi': 'casadi is not installed'}
    assert list(report['models'][0]['backends']) == ['myokit']
    assert 'Skipped: casadi (casadi is not installed).' in table


@pytest.mark.unit
def test_unknown_backend_is_an_argument_error():
    with pytest.raises(SystemExit):
        solver_comparison.parse_args(['--backends', 'opencor'])
