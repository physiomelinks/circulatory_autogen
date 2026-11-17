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
                     with_samples=True, with_jacobian=True, bounds=None, jacobian_points=8):
    """A Sobol run of ``outputs_of`` in ``directory``, in sobolSA's file layout, with the local
    Jacobian sobolSA.run_local_jacobian saves (forward differences at the first
    ``jacobian_points`` Saltelli base points), made by sobolSA's own methods."""
    os.makedirs(directory, exist_ok=True)
    bounds = bounds or [[1.0, 2.0]] * len(params)
    problem = {'num_vars': len(params), 'names': params, 'bounds': bounds}
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
        columns = list(outputs)
        values = np.column_stack([outputs[k][0] for k in columns])
        pd.DataFrame(samples, columns=params).to_csv(f'{stem}_samples.csv', index=False)
        pd.DataFrame(values, columns=columns).to_csv(f'{stem}_outputs.csv', index=False)
        meta = {}
        if with_jacobian:
            sa = _bare_sobol_sa(*zip(*bounds), lambda x: [outputs_of(x[None, :])[k][0][0]
                                                           for k in columns])
            sa.jacobian_points = jacobian_points
            jacobian = sa.run_local_jacobian(samples, values)
            rows = sa.local_jacobian_rows
            frame = pd.DataFrame(jacobian.reshape(-1, len(columns)), columns=columns)
            frame.insert(0, 'Parameter', params * len(rows))
            frame.insert(0, 'point', np.repeat(rows, len(params)))
            frame.to_csv(f'{stem}_local_jacobian.csv', index=False)
            meta = {'local_jacobian': {'points': rows.tolist(), 'log_step': 1e-2}}
        with open(os.path.join(directory, 'sobol_problem.json'), 'w') as f:
            json.dump({'names': params, 'bounds': bounds, 'num_samples': n,
                       'sample_type': 'saltelli', 'calc_second_order': True, **meta,
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
               'variance_fraction': 0.9, 'min_index': 0.1, 'redundancy_check': False}
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


TIMES = np.linspace(0.0, 6.0, 30)


def _time_course(x):
    """A time course in which only a / c and b matter: (a / c) sin(b t), std 0.05."""
    a, b, c = x.T
    return {f'y(t={t:.2f})': ((a / c) * np.sin(b * t), 0.05) for t in TIMES}


@pytest.mark.unit
def test_a_time_course_keeps_one_parameter_of_a_ratio_with_the_redundancy_check(tmp_path):
    params = ['m/a', 'm/b', 'm/c']
    run = _write_sobol_run(str(tmp_path / 'sa'), outputs_of=_time_course, params=params)
    source = _params_for_id(str(tmp_path / 'params_for_id.json'), params)

    kept = PCASobolReduction().reduce(run, source, str(tmp_path / 'pca' / 'kept.json'))
    names = [row['parameter'] for row in kept]
    # b, and one of a and c: a / c is one direction however many modes it drives
    assert 'm/b' in names and len(names) == 2
    report = json.loads((tmp_path / 'pca' / 'param_reduction.json').read_text())
    left = [n for n in ('m/a', 'm/c') if n not in names][0]
    assert report['unexplained_effects'][left] < 1e-6 * report['effects'][left]
    assert report['n_informative_modes'] > len(params)

    # one parameter per informative mode keeps all three: there are more modes than parameters
    every = PCASobolReduction(redundancy_check=False).reduce(
        run, source, str(tmp_path / 'per_mode' / 'kept.json'))
    assert {row['parameter'] for row in every} == set(params)


@pytest.mark.unit
def test_the_redundancy_check_needs_the_local_jacobian(tmp_path):
    run = _write_sobol_run(str(tmp_path / 'sa'), with_jacobian=False)
    source = _params_for_id(str(tmp_path / 'params_for_id.json'))
    with pytest.raises(SensitivityFilesError, match='local_jacobian'):
        PCASobolReduction().reduce(run, source, str(tmp_path / 'out' / 'kept.json'))
    assert PCASobolReduction(redundancy_check=False).reduce(
        run, source, str(tmp_path / 'out' / 'kept.json'))



def _bare_sobol_sa(mins, maxs, model):
    """A sobol_SA over ``mins``/``maxs`` whose simulations are ``model(point)`` on one process,
    without building a model."""
    from libcuflynx.sensitivity_analysis.sobolSA import sobol_SA
    sa = sobol_SA.__new__(sobol_SA)
    sa.SA_info = {'param_mins': list(mins), 'param_maxs': list(maxs)}
    sa.rank, sa.num_procs, sa.use_mpi, sa.jacobian_points = 0, 1, False, 8
    sa.generate_outputs_mpi = lambda points: np.array([model(p) for p in points])
    return sa


@pytest.mark.unit
def test_the_local_jacobian_is_taken_at_saltelli_base_points_scaled_to_each_range():
    """One forward step per parameter at each of the first jacobian_points base points (the
    rows starting each block of 2M + 2), each derivative times the parameter's range: in
    ln(parameter), or, for bounds that straddle 0, in the parameter itself."""
    mins, maxs = [-1.0, 0.5, -3.0], [1.0, 2.0, 1.0]
    sa = _bare_sobol_sa(mins, maxs, lambda p: [3 * p[0] + 5 * p[2], np.log(p[1])])
    sa.jacobian_points = 4
    samples = saltelli.sample({'num_vars': 3, 'names': list('abc'),
                               'bounds': list(zip(mins, maxs))}, 16, calc_second_order=True)
    outputs = np.array([[3 * p[0] + 5 * p[2], np.log(p[1])] for p in samples])
    jacobian = sa.run_local_jacobian(samples, outputs)
    assert list(sa.local_jacobian_rows) == [0, 8, 16, 24]
    assert jacobian.shape == (4, 3, 2)
    for block in jacobian:   # linear in p0, p2 and in ln p1: the same at every point
        assert block[0] == pytest.approx([3 * 2, 0], abs=1e-9)        # 3 * (1 - -1)
        assert block[1] == pytest.approx([0, np.log(4)], abs=1e-9)    # ln 2 - ln 0.5
        assert block[2] == pytest.approx([5 * 4, 0], abs=1e-9)        # 5 * (1 - -3)


@pytest.mark.unit
def test_a_sobol_run_on_one_process_saves_the_local_jacobian():
    """The redundancy check, on by default, needs the local Jacobian, so every Sobol run makes
    it, with or without use_MPI, from the run's own outputs at the base points."""
    sa = _bare_sobol_sa([0.5, 0.5], [2.0, 2.0], lambda p: [p[0] * p[1]])
    samples = np.array([[1.0, 1.0]] * 6 + [[2.0, 0.5]] * 6)
    sa.generate_samples = lambda: samples
    sa.sobol_index = lambda outputs: ('S1', 'ST', 'S2')
    assert sa.run() == ('S1', 'ST', 'S2')
    assert list(sa.local_jacobian_rows) == [0, 6]
    # d(p0 p1)/d ln p0 = p0 p1, times ln 4; forward step: off by h/2
    assert sa.local_jacobian[:, 0, 0] == pytest.approx([np.log(4), np.log(4)], rel=1e-2)


@pytest.mark.unit
@pytest.mark.parametrize('value', [-1, 2.5, True, 'eight'])
def test_jacobian_points_is_a_whole_number(value):
    from libcuflynx.sensitivity_analysis.sobolSA import jacobian_points_option
    assert jacobian_points_option({}) == 8 and jacobian_points_option({'jacobian_points': 0}) == 0
    with pytest.raises(ValueError, match='jacobian_points'):
        jacobian_points_option({'jacobian_points': value})


@pytest.mark.unit
def test_a_parameter_flat_at_the_centre_of_its_range_is_kept(tmp_path):
    """p^2 on [-1, 1] has slope 0 at the centre of its box but moves the data far above the
    noise elsewhere; derivatives at several points see it. a and b enter only as a * b, so one
    of them is kept."""
    params = ['m/p', 'm/a', 'm/b']

    def outputs(x):
        p, a, b = x.T
        return {'y1': (p ** 2, 0.05), 'y2': (a * b, 0.05), 'y3': (a * b + p ** 2, 0.1)}

    run = _write_sobol_run(str(tmp_path / 'sa'), outputs_of=outputs, params=params,
                           bounds=[[-1.0, 1.0], [1.0, 2.0], [1.0, 2.0]])
    source = _params_for_id(str(tmp_path / 'params_for_id.json'), params=params)
    kept = [r['parameter'] for r in PCASobolReduction().reduce(
        run, source, str(tmp_path / 'pca' / 'kept.json'))]
    assert 'm/p' in kept and len(kept) == 2 and len({'m/a', 'm/b'} & set(kept)) == 1


@pytest.mark.unit
def test_a_redundancy_whose_coefficients_vary_over_the_box_keeps_one_parameter(tmp_path):
    """The data depend on a + b only. In ln(parameter), b's effect is a's times b / a, which
    changes from point to point: explained at each point, so one of a and b is kept, although
    the two effects stacked over the points are not parallel."""
    params = ['m/a', 'm/b', 'm/c']

    def outputs(x):
        a, b, c = x.T
        return {f'y(t={t:.2f})': ((a + b) * np.exp(-c * t), 0.05) for t in TIMES}

    run = _write_sobol_run(str(tmp_path / 'sa'), outputs_of=outputs, params=params)
    source = _params_for_id(str(tmp_path / 'params_for_id.json'), params=params)
    kept = [r['parameter'] for r in PCASobolReduction().reduce(
        run, source, str(tmp_path / 'pca' / 'kept.json'))]
    assert 'm/c' in kept and len(kept) == 2 and len({'m/a', 'm/b'} & set(kept)) == 1
