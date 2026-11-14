"""Parameter reduction from a Sobol analysis (sensitivity_analysis/parameter_reduction.py).

The Sobol runs here are of analytic functions, sampled and analysed with SALib directly and
saved in the layout sobolSA writes, so no model is simulated.
"""

import json
import os

import numpy as np
import pandas as pd
import pytest
from SALib.analyze import sobol
from SALib.sample import saltelli

from libcuflynx.parsers.PrimitiveParsers import analysis_options
from libcuflynx.sensitivity_analysis.parameter_reduction import (
    PATH_OPTIONS, ParamReduction, PCASobolReduction, SobolReduction, SensitivityFilesError,
    run_param_reduction, sobol_run_files)
from libcuflynx.utilities.params_for_id_file import write_params_for_id

PARAMS = ['m/A', 'm/B', 'm/C', 'm/D']


def _toy_outputs(x):
    """A and B enter only as their product; C on its own; D far below the noise."""
    a, b, c, d = x.T
    return {'y1': (a * b, 0.05), 'y2': (2 * a * b + c, 0.1), 'y3': (c, 0.05),
            'y4': (0.01 * d, 0.1)}


def _write_sobol_run(directory, outputs_of=_toy_outputs, params=PARAMS, n=256, std_scale=1.0,
                     with_samples=True):
    """A Sobol run of ``outputs_of`` in ``directory``, in sobolSA's file layout."""
    os.makedirs(directory, exist_ok=True)
    problem = {'num_vars': len(params), 'names': params, 'bounds': [[1.0, 2.0]] * len(params)}
    samples = saltelli.sample(problem, n, calc_second_order=True)
    outputs = outputs_of(samples)
    stem = os.path.join(directory, f'all_outputs_n{n}_Sobol')
    table = pd.DataFrame({'Parameter': params})
    for name, (values, _) in outputs.items():
        result = sobol.analyze(dict(problem, names=pd.Series(params)), values,
                               calc_second_order=True)
        table[f'S1_{name}'], table[f'ST_{name}'] = result['S1'], result['ST']
    table.to_csv(f'{stem}_indices.csv', index=False)
    if with_samples:
        pd.DataFrame(samples, columns=params).to_csv(f'{stem}_samples.csv', index=False)
        pd.DataFrame({k: v for k, (v, _) in outputs.items()}).to_csv(f'{stem}_outputs.csv',
                                                                     index=False)
        with open(os.path.join(directory, 'sobol_problem.json'), 'w') as f:
            json.dump({'names': params, 'bounds': problem['bounds'], 'num_samples': n,
                       'sample_type': 'saltelli', 'calc_second_order': True,
                       'outputs': [{'output': k, 'kind': 'data_item',
                                    'std': None if std is None else std * std_scale,
                                    'weight': 1.0} for k, (_, std) in outputs.items()]}, f)
    return directory


def _params_for_id(path, params=PARAMS):
    write_params_for_id(path, [{'targets': [p], 'min': 1.0, 'max': 2.0,
                                'name_for_plotting': p.split('/')[1]} for p in params])
    return path


@pytest.mark.unit
def test_pca_keeps_one_of_two_parameters_that_move_the_data_together(tmp_path):
    run = _write_sobol_run(str(tmp_path / 'sa'))
    source = _params_for_id(str(tmp_path / 'params_for_id.json'))

    pca = PCASobolReduction()
    kept = pca.reduce(run, source, str(tmp_path / 'pca' / 'kept.json'))
    names = [row['parameter'] for row in kept]
    # two informative modes: A*B and C. One of A and B, never both; never D.
    assert len(names) == 2 and 'm/C' in names
    assert len({'m/A', 'm/B'} & set(names)) == 1
    assert [row['mode'] for row in kept] == sorted(row['mode'] for row in kept)
    report = json.loads((tmp_path / 'pca' / 'param_reduction.json').read_text())
    assert report['n_informative_modes'] == 2
    for name in ('pca_modes.csv', 'pca_mode_sobol_indices.csv'):
        assert (tmp_path / 'pca' / name).is_file()

    # Top-N by max ST over the raw outputs keeps D, which alone drives y4 although y4 moves
    # by a tenth of its noise, ahead of A*B, the largest effect on the data.
    simple = [row['parameter'] for row in SobolReduction(n_params=2).reduce(
        run, source, str(tmp_path / 'simple' / 'kept.json'))]
    assert set(simple) == {'m/C', 'm/D'}


@pytest.mark.unit
def test_a_mode_inside_the_noise_is_not_informative(tmp_path):
    source = _params_for_id(str(tmp_path / 'params_for_id.json'))
    quiet = _write_sobol_run(str(tmp_path / 'quiet'), std_scale=1e3)
    with pytest.raises(ValueError, match='kept no parameter'):
        PCASobolReduction().reduce(quiet, source, str(tmp_path / 'out' / 'kept.json'))
    # the noise floor is the option: lowered, the same modes count again
    kept = PCASobolReduction(min_mode_std=1e-4).reduce(quiet, source,
                                                       str(tmp_path / 'out' / 'kept.json'))
    assert 'm/C' in [row['parameter'] for row in kept]


@pytest.mark.unit
def test_with_one_output_the_mode_indices_are_the_output_indices(tmp_path):
    run = _write_sobol_run(str(tmp_path / 'sa'),
                           outputs_of=lambda x: {'y': (x[:, 0] + 0.5 * x[:, 2] ** 2, 0.01)})
    files = sobol_run_files(run)
    PCASobolReduction().select(files, str(tmp_path))
    modes = pd.read_csv(tmp_path / 'pca_mode_sobol_indices.csv').set_index('Parameter')
    raw = pd.read_csv(files['indices']).set_index('Parameter')
    # a mode score is an affine function of the one output, so the indices are the same
    assert modes['ST_mode_1'].to_numpy() == pytest.approx(raw['ST_y'].to_numpy(), abs=1e-9)
    assert modes['S1_mode_1'].to_numpy() == pytest.approx(raw['S1_y'].to_numpy(), abs=1e-9)


@pytest.mark.unit
def test_a_pca_reduction_needs_the_saved_samples(tmp_path):
    run = _write_sobol_run(str(tmp_path / 'sa'), with_samples=False)
    source = _params_for_id(str(tmp_path / 'params_for_id.json'))
    with pytest.raises(SensitivityFilesError, match='rerun the Sobol analysis'):
        PCASobolReduction().reduce(run, source, str(tmp_path / 'out' / 'kept.json'))
    # the sobol method reads only the indices
    assert SobolReduction(n_params=1).reduce(run, source, str(tmp_path / 'out' / 'kept.json'))


@pytest.mark.unit
def test_options_are_the_param_reduction_options_descriptors():
    descriptors = {o['name']: o for o in analysis_options('param_reduction')}
    # every option is read by a method, or says where to read and write
    read = set(PCASobolReduction.OPTIONS) | set(SobolReduction.OPTIONS) | {'method'}
    assert set(descriptors) == read | set(PATH_OPTIONS)
    assert descriptors['method']['choices'] == ['pca', 'sobol']
    # each method's defaults are the descriptors'
    pca = ParamReduction.from_options({})
    assert isinstance(pca, PCASobolReduction)
    for name in PCASobolReduction.OPTIONS:
        assert getattr(pca, name) == descriptors[name]['default']
    sobol_defaults = ParamReduction.from_options({'method': 'sobol', 'n_params': 2}).as_dict()
    for name in ('index_type', 'criterion', 'threshold'):
        assert sobol_defaults[name] == descriptors[name]['default']
    # round trip, and the paths are ignored
    options = {'method': 'pca', 'n_params': 3, 'index_type': 'S1', 'min_mode_std': 2.0,
               'variance_fraction': 0.9, 'min_index': 0.1}
    assert ParamReduction.from_options(dict(options, sensitivity_dir='x')).as_dict() == options
    # an option of the other method at its default is fine (a filled-in form); changed, not
    assert ParamReduction.from_options({'method': 'pca', 'criterion': 'max'})
    with pytest.raises(ValueError, match="criterion is not read by method 'pca'"):
        ParamReduction.from_options({'method': 'pca', 'criterion': 'mean'})
    with pytest.raises(ValueError, match='has no'):
        ParamReduction.from_options({'top_n': 3})
    with pytest.raises(ValueError, match='method is one of'):
        ParamReduction.from_options({'method': 'svd'})


@pytest.mark.unit
def test_run_param_reduction_writes_into_its_output_dir(tmp_path):
    run = _write_sobol_run(str(tmp_path / 'sa'))
    source = _params_for_id(str(tmp_path / 'params_for_id.json'))
    inp = {'sa_options': {'output_dir': run}, 'params_for_id_path': source,
           'param_reduction_options': {'method': 'pca'}}
    kept = run_param_reduction(inp)
    out = tmp_path / 'sa' / 'param_reduction'
    assert (out / 'reduced_params_for_id.json').is_file()
    assert json.loads((out / 'param_reduction.json').read_text())['selected'] == kept


_RAISING_REDUCTION_SCRIPT = '''
import sys
from libcuflynx.parsers.PrimitiveParsers import YamlFileParser
from libcuflynx.sensitivity_analysis import parameter_reduction
from libcuflynx.sensitivity_analysis.sensitivityAnalysis import SensitivityAnalysis
from libcuflynx.scripts import sensitivity_analysis_run_script as script

class _Agent:
    def run_sensitivity_analysis(self):
        pass

def _boom(inp):
    raise ValueError('reduction failed on rank 0')

YamlFileParser.parse_user_inputs_file = lambda self, inp, **kw: inp
SensitivityAnalysis.init_from_dict = staticmethod(lambda inp: _Agent())
parameter_reduction.run_param_reduction = _boom
rank = script.MPI.COMM_WORLD.Get_rank()
try:
    script.run_SA({'do_param_reduction': True})
except ValueError as exc:
    print(f'RANK{rank} RAISED {exc}', flush=True)
    sys.exit(0)
print(f'RANK{rank} NO ERROR', flush=True)
'''


@pytest.mark.unit
def test_a_failing_reduction_raises_on_every_rank_instead_of_hanging_them(tmp_path):
    """run_SA reduces on rank 0; if that raises, the other ranks must raise too, not wait at
    a barrier for a rank that has left. A hang fails the test by the timeout."""
    import subprocess
    import sys
    script = tmp_path / 'run.py'
    script.write_text(_RAISING_REDUCTION_SCRIPT)
    env = dict(os.environ, OMP_NUM_THREADS='1')
    try:
        proc = subprocess.run(['mpiexec', '-n', '2', sys.executable, str(script)], env=env,
                              capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        pytest.fail('a rank hung after the reduction failed on rank 0')
    for rank in (0, 1):
        assert f'RANK{rank} RAISED reduction failed on rank 0' in proc.stdout, \
            proc.stdout + proc.stderr


@pytest.mark.unit
def test_default_has_no_relative_cut_so_a_resolved_small_mode_is_kept(tmp_path):
    # C moves the data 5.8 noise stds but is ~0.2% of the scaled variance next to A*B
    params = ['m/A', 'm/B', 'm/C']

    def outputs(x):
        a, b, c = x.T
        return {'y1': (a * b, 0.005), 'y3': (c, 0.05)}

    run = _write_sobol_run(str(tmp_path / 'sa'), outputs_of=outputs, params=params)
    source = _params_for_id(str(tmp_path / 'params_for_id.json'), params=params)
    assert PCASobolReduction().variance_fraction == 1.0
    kept = [r['parameter'] for r in PCASobolReduction().reduce(
        run, source, str(tmp_path / 'default' / 'kept.json'))]
    assert 'm/C' in kept and len({'m/A', 'm/B'} & set(kept)) == 1
    cut = [r['parameter'] for r in PCASobolReduction(variance_fraction=0.99).reduce(
        run, source, str(tmp_path / 'cut' / 'kept.json'))]
    assert 'm/C' not in cut and len(cut) == 1


@pytest.mark.unit
def test_top_n_keeps_both_parameters_of_a_product_and_pca_keeps_one(tmp_path):
    # E adds a small term to y4: above its noise, but ST_E far below A's and B's.
    params = ['m/A', 'm/B', 'm/C', 'm/E']

    def outputs(x):
        a, b, c, e = x.T
        return {'y1': (a * b, 0.05), 'y2': (2 * a * b + c, 0.1), 'y3': (c, 0.05),
                'y4': (c + 0.5 * e, 0.1)}

    run = _write_sobol_run(str(tmp_path / 'sa'), outputs_of=outputs, params=params)
    source = _params_for_id(str(tmp_path / 'params_for_id.json'), params=params)
    # top-3 by max ST keeps both A and B (each ranks above E) and C, leaving E out
    top = [r['parameter'] for r in SobolReduction(n_params=3).reduce(
        run, source, str(tmp_path / 'top' / 'kept.json'))]
    assert set(top) == {'m/A', 'm/B', 'm/C'}
    # PCA sees A and B as one mode (A*B), keeps C, and keeps E, whose term in y4 is a mode of its
    # own above the noise floor (mode std 1.3 against 1): one of A and B, C and E.
    pca = [r['parameter'] for r in PCASobolReduction().reduce(
        run, source, str(tmp_path / 'pca' / 'kept.json'))]
    assert len(pca) == 3 and {'m/C', 'm/E'} <= set(pca) and len({'m/A', 'm/B'} & set(pca)) == 1
