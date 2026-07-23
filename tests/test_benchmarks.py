"""Unit tests for the benchmark scaling harness: the MPI-free registry stays in sync with the
run registry, and the core-scaling Markdown renders correctly. These do not run any benchmark."""
import os
import sys

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
                                    params=[1.0, 2.0]))
    result.rows.append(BenchmarkRow(method='m2', skipped_reason='no licence'))
    d = benchmark_result_to_dict(result)
    assert d['name'] == 'b' and d['true_params'] == [1.0, 2.0]
    assert d['rows'][0] == {'method': 'm1', 'cost': 1.5e-3, 'time_s': 12.0,
                            'param_err': 0.01, 'skipped_reason': None}
    assert d['rows'][1]['skipped_reason'] == 'no licence'


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
