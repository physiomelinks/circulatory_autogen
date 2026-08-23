"""Unit tests for the benchmark scaling harness: the MPI-free registry stays in sync with the
run registry, and the core-scaling Markdown renders correctly. These do not run any benchmark."""
import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from benchmarks.registry import BENCHMARK_CI
from benchmarks.docs_results import (
    BenchmarkResult, BenchmarkRow, ScalingBenchmarkResult, ScalingRow,
    benchmark_result_to_dict, scaling_result_to_markdown)


@pytest.mark.unit
def test_registry_matches_benchmark_specs():
    """The MPI-free registry (used by the scaling orchestrator) must list exactly the benchmarks
    in benchmark_specs.BENCHMARKS, with matching CI flags -- otherwise the orchestrator selects a
    different set than the runner."""
    from benchmarks.benchmark_specs import BENCHMARKS
    assert set(BENCHMARK_CI) == set(BENCHMARKS), (set(BENCHMARK_CI), set(BENCHMARKS))
    for name, spec in BENCHMARKS.items():
        assert BENCHMARK_CI[name] == spec['ci'], name


@pytest.mark.unit
def test_benchmark_result_to_dict_roundtrips_rows():
    result = BenchmarkResult(
        name='b', title='B', description='d', env_note='e',
        true_params=[1.0, 2.0], param_labels=['a', 'b'])
    result.rows.append(BenchmarkRow(method='m1', cost=1.5e-3, time_s=12.0, param_err=0.01,
                                    params=[1.0, 2.0], evals=20000))
    result.rows.append(BenchmarkRow(method='m2', skipped_reason='no licence'))
    d = benchmark_result_to_dict(result)
    assert d['name'] == 'b' and d['true_params'] == [1.0, 2.0]
    # `evals` travels with the row: a scaling child hands its numbers back through this dict,
    # and the orchestrator needs the evaluation count to say whether the core counts did equal
    # work (#344). A row whose optimiser does not report one carries None.
    assert d['rows'][0] == {'method': 'm1', 'cost': 1.5e-3, 'time_s': 12.0,
                            'param_err': 0.01, 'evals': 20000, 'skipped_reason': None}
    assert d['rows'][1]['skipped_reason'] == 'no licence'
    assert d['rows'][1]['evals'] is None


@pytest.mark.unit
def test_scaling_result_to_markdown_has_a_column_per_core():
    result = ScalingBenchmarkResult(
        name='fitzhugh_nagumo', title='FHN', description='desc', cores=[1, 2, 4],
        env_note='cores: 1, 2, 4', true_params=[0.2, 0.2, 3.0], param_labels=['a', 'b', 'c'])
    result.rows.append(ScalingRow(method='genetic_algorithm', cost=1.2e-3, param_err=0.03,
                                  times_by_core={1: 40.0, 2: 21.0, 4: 11.5}))
    # A method missing a core count shows a dash there.
    result.rows.append(ScalingRow(method='multi_start (CasADi AD)', cost=3.4e-9, param_err=0.001,
                                  times_by_core={1: 30.0, 4: 8.2}))
    result.rows.append(ScalingRow(method='multi_start (AADC AD)', skipped_reason='no licence'))

    md = scaling_result_to_markdown(result)
    header = next(ln for ln in md.splitlines() if ln.startswith('| method'))
    assert '1 core (s)' in header and '2 cores (s)' in header and '4 cores (s)' in header
    assert 'max param err' in header

    ga = next(ln for ln in md.splitlines() if ln.startswith('| `genetic_algorithm`'))
    assert '40.0' in ga and '21.0' in ga and '11.5' in ga
    # missing 2-core cell renders as an em dash
    ad = next(ln for ln in md.splitlines() if ln.startswith('| `multi_start (CasADi AD)`'))
    assert '30.0' in ad and '8.2' in ad and '—' in ad
    # skipped row is rendered and carries its reason
    assert any('skipped' in ln and 'no licence' in ln for ln in md.splitlines())
    assert 'True parameters: a=0.2, b=0.2, c=3.' in md


# ------------------------------------------------------------------------------------------
# CUFLynx paper table generation (benchmarks/create_CUFLynx_paper_tables.py)
# ------------------------------------------------------------------------------------------

from benchmarks.create_CUFLynx_paper_tables import (  # noqa: E402
    best_row, load_results, load_scaling, parse_markdown_results, slowest_benchmark,
    table_calibration, table_scaling, tex_escape)
from benchmarks.docs_results import results_to_markdown  # noqa: E402


def _example_result():
    result = BenchmarkResult(
        name='goodwin', title='Goodwin oscillator (external PMR CellML, non-stiff, multimodal)',
        description='desc', env_note='8 MPI rank(s); 30000 cost evaluations',
        true_params=[72.0, 2.0, 36.0], param_labels=['a_i', 'b_i', 'A_i'])
    result.rows.append(BenchmarkRow(method='genetic_algorithm', cost=1.0099e-3, time_s=41.1,
                                    param_err=1.4934))
    result.rows.append(BenchmarkRow(method='multi_start (FD)', cost=9.1452e-15, time_s=15.0,
                                    param_err=0.0))
    result.rows.append(BenchmarkRow(method='multi_start (AADC AD)', skipped_reason='no licence'))
    return result


@pytest.mark.unit
def test_markdown_results_roundtrip_into_the_table_generator(tmp_path):
    """The paper tables are built by parsing the harness's own Markdown, so a format change in
    docs_results must not silently produce empty/incorrect tables."""
    md_path = tmp_path / 'result_goodwin.md'
    md_path.write_text(results_to_markdown([_example_result()], generated_note='note'))

    parsed = parse_markdown_results(str(md_path))
    assert len(parsed) == 1
    result = parsed[0]
    assert result['name'] == 'goodwin'
    assert result['true_params'] == [72.0, 2.0, 36.0]

    by_method = {r['method']: r for r in result['rows']}
    assert by_method['genetic_algorithm']['cost'] == pytest.approx(1.0099e-3)
    assert by_method['genetic_algorithm']['time_s'] == pytest.approx(41.1)
    assert by_method['genetic_algorithm']['param_err'] == pytest.approx(1.4934)
    assert by_method['multi_start (FD)']['param_err'] == 0.0
    # a skipped optimiser round-trips as a skip, not as a zero-time row
    assert by_method['multi_start (AADC AD)']['skipped_reason'] == 'no licence'
    assert by_method['multi_start (AADC AD)']['time_s'] is None


@pytest.mark.unit
def test_best_row_prefers_lowest_param_error_then_time():
    parsed = {'rows': [
        {'method': 'slow_exact', 'cost': 1e-9, 'time_s': 500.0, 'param_err': 0.0,
         'skipped_reason': None},
        {'method': 'fast_exact', 'cost': 1e-8, 'time_s': 10.0, 'param_err': 0.0,
         'skipped_reason': None},
        {'method': 'fast_wrong', 'cost': 1e-12, 'time_s': 1.0, 'param_err': 1.5,
         'skipped_reason': None},
        {'method': 'skipped', 'cost': None, 'time_s': None, 'param_err': None,
         'skipped_reason': 'no licence'},
    ]}
    # lowest cost belongs to fast_wrong, but it recovers the wrong parameters
    assert best_row(parsed)['method'] == 'fast_exact'


@pytest.mark.unit
def test_slowest_benchmark_uses_total_wall_clock():
    by_name = {
        'cheap': {'rows': [{'method': 'a', 'cost': 1.0, 'time_s': 10.0, 'param_err': 0.0,
                            'skipped_reason': None}]},
        'pricey': {'rows': [{'method': 'a', 'cost': 1.0, 'time_s': 900.0, 'param_err': 0.0,
                             'skipped_reason': None},
                            {'method': 'b', 'cost': 1.0, 'time_s': 2800.0, 'param_err': 0.0,
                             'skipped_reason': None}]},
    }
    assert slowest_benchmark(by_name) == 'pricey'


@pytest.mark.unit
def test_calibration_table_escapes_latex_and_reports_one_row_per_model(tmp_path):
    md_path = tmp_path / 'result_goodwin.md'
    md_path.write_text(results_to_markdown([_example_result()]))
    by_name = load_results([str(md_path)])

    tex = table_calibration(by_name, list(by_name))
    # underscores in optimiser names must be escaped or LaTeX fails to compile
    assert r'\texttt{multi\_start (FD)}' in tex
    assert 'genetic' not in tex          # only the best optimiser in the default (per-model) view
    assert r'\begin{table}' in tex and r'\bottomrule' in tex
    assert 'Goodwin oscillator' in tex
    assert ' 3 &' in tex                 # parameter count comes from true_params

    tex_all = table_calibration(by_name, list(by_name), all_methods=True)
    assert r'\texttt{genetic\_algorithm}' in tex_all
    # a skipped optimiser contributes no row
    assert 'AADC' not in tex_all


@pytest.mark.unit
def test_scaling_table_reads_the_per_core_cache_and_computes_speedup(tmp_path):
    jdir = tmp_path / 'three_compartment'
    jdir.mkdir()
    for cores, factor in ((1, 4.0), (2, 2.0), (4, 1.0)):
        payload = {'num_ranks': cores, 'result': {
            'name': 'three_compartment', 'title': '3compartment cardiovascular (stiff)',
            'description': 'd', 'env_note': 'e', 'true_params': [1.0], 'param_labels': ['p'],
            'rows': [
                {'method': 'genetic_algorithm', 'cost': 2.2e-2, 'time_s': 100.0 * factor,
                 'param_err': 0.3, 'skipped_reason': None},
                {'method': 'multi_start (AADC AD)', 'cost': None, 'time_s': None,
                 'param_err': None, 'skipped_reason': 'no licence'},
            ]}}
        (jdir / f'scaling_{cores}core.json').write_text(json.dumps(payload))

    per_core = load_scaling('three_compartment', str(tmp_path))
    assert sorted(per_core) == [1, 2, 4]

    tex = table_scaling({'name': 'three_compartment', 'title': '3compartment'}, per_core)
    ga = next(ln for ln in tex.splitlines() if 'genetic' in ln)
    assert '400' in ga and '200' in ga and '100' in ga
    assert '4.00$\\times$' in ga          # 400s at 1 core -> 100s at 4 cores
    # a method skipped at every core count is omitted rather than rendered blank
    assert 'AADC' not in tex


@pytest.mark.unit
def test_tex_escape_covers_the_specials_that_appear_in_method_names():
    assert tex_escape('multi_start (FD) & 100%') == r'multi\_start (FD) \& 100\%'

@pytest.mark.unit
def test_three_compartment_ground_truth_is_inside_the_identification_bounds():
    """The synthetic ground truth must be recoverable, i.e. inside the search box.

    The model's own q_lv_init default (2e-3) is OUTSIDE 3compartment_params_for_id.csv's bounds
    of [2e-4, 1.5e-3], so the CSV defaults cannot be used wholesale as a ground truth -- an
    optimiser could never reach it and the reported error would never go to zero.
    """
    from benchmarks.benchmark_specs import (
        THREE_COMPARTMENT_TRUE_PARAMS, THREE_COMPARTMENT_PARAM_LABELS,
        THREE_COMPARTMENT_PARAM_NAMES)

    bounds = {  # from resources/3compartment_params_for_id.csv
        'global/q_lv_init': (200e-6, 1500e-6),
        'aortic_root/C': (1e-9, 5e-8),
        'global/E_lv_A': (1e8, 5e8),
        'global/E_lv_B': (1e6, 5e7),
    }
    assert len(THREE_COMPARTMENT_TRUE_PARAMS) == len(THREE_COMPARTMENT_PARAM_LABELS)
    assert len(THREE_COMPARTMENT_TRUE_PARAMS) == len(THREE_COMPARTMENT_PARAM_NAMES)
    for name, value in zip(THREE_COMPARTMENT_PARAM_NAMES, THREE_COMPARTMENT_TRUE_PARAMS):
        lo, hi = bounds[name]
        assert lo < value < hi, f'{name} ground truth {value:g} outside bounds [{lo:g}, {hi:g}]'


@pytest.mark.unit
def test_obs_reducers_cover_the_shipped_operations():
    """Every operation in the 3compartment obs template must have a reducer, or the synthetic
    ground truth silently cannot be generated for that observable."""
    import json as _json
    from benchmarks.benchmark_specs import _OBS_REDUCERS

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, 'resources', '3compartment_obs_data.json')) as f:
        items = _json.load(f)
    missing = sorted({it['operation'] for it in items} - set(_OBS_REDUCERS))
    assert not missing, f'no reducer for obs operation(s): {missing}'

    y = np.array([1.0, 5.0, 3.0])
    assert _OBS_REDUCERS['mean'](y) == pytest.approx(3.0)
    assert _OBS_REDUCERS['max'](y) == pytest.approx(5.0)
    assert _OBS_REDUCERS['min'](y) == pytest.approx(1.0)
    assert _OBS_REDUCERS['max_minus_min'](y) == pytest.approx(4.0)


@pytest.mark.unit
def test_every_benchmark_reports_relative_parameter_error():
    """The max param err column must mean the same thing in every row.

    Parameters within one model span orders of magnitude (3compartment ~8e-4 to ~3.7e8, Teusink
    226 to 1185), so an absolute maximum reports only the largest parameter. It also makes the
    column incomparable between benchmarks -- 0.01 on FitzHugh-Nagumo and 0.01 on 3compartment
    would describe different qualities of fit. FitzHugh-Nagumo and Goodwin previously reported
    absolute error while Teusink and 3compartment reported relative.
    """
    import inspect
    from benchmarks import benchmark_specs
    from benchmarks.benchmark_specs import max_relative_param_err

    # a 10% miss is 0.1 whatever the parameter's magnitude
    assert max_relative_param_err([1.1], [1.0]) == pytest.approx(0.1)
    assert max_relative_param_err([1.1e8], [1.0e8]) == pytest.approx(0.1)
    # the largest *relative* miss wins, not the largest absolute one
    assert max_relative_param_err([1.01, 2.0e8], [1.0, 1.0e8]) == pytest.approx(1.0)

    # no benchmark computes its own error any more
    src = inspect.getsource(benchmark_specs)
    assert "param_err=float(np.max(np.abs(" not in src, (
        "a benchmark is still computing param_err by hand; use max_relative_param_err")
    assert src.count("param_err=max_relative_param_err(") == 4, (
        "all four benchmarks should report error through the shared helper")
