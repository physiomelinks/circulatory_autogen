"""Compare the PCA reduction's redundancy-check variants on the study models, under MPI::

    mpiexec -n 4 python compare_redundancy.py <study> <out> [--models goodwin ...]
        [--num-samples 256] [--variants centre_central_1e-2 ...]

``<study>`` is a finished study (run_study.py per model), whose ``inputs``, ``synthetic`` and
``generated_models`` each model reuses and whose ``compare/summary.json`` gives the reference:
the number of eigenvalues of at least 1 of the Fisher information at the truth.

For each model and derivative variant (DERIVATIVE_VARIANTS: where the local derivatives are
taken, the difference scheme and step), one Sobol run through ``run_SA`` into
``<out>/<model>/<variant>/sensitivity``. Every statistic (STATISTICS: how a parameter's
unexplained effect is summarised over the points, and whether unconverged points are left
out) then reduces that run's saved derivatives, with no new simulations. A finished run or
reduction is skipped, so a stopped comparison resumes.

Writes ``<out>/redundancy_comparison.json`` (per model, variant and statistic: the kept
parameters, their count and each dropped parameter's unexplained effect) and
``<out>/redundancy_comparison.md``, a table of the counts against the Fisher count.
"""

import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import SPECS, paths, raw_inputs  # noqa: E402

#: sa_options of each derivative variant.
DERIVATIVE_VARIANTS = {
    'centre_central_1e-2': {'jacobian_location': 'centre', 'jacobian_step': 1e-2},
    'K8_forward_1e-2': {'jacobian_points': 8, 'jacobian_scheme': 'forward',
                        'jacobian_step': 1e-2},
    'K8_central_1e-2': {'jacobian_points': 8, 'jacobian_step': 1e-2},
    'K8_central_1e-3_checked': {'jacobian_points': 8, 'jacobian_step': 1e-3,
                                'jacobian_check_steps': True},
    'K16_central_1e-3_checked': {'jacobian_points': 16, 'jacobian_step': 1e-3,
                                 'jacobian_check_steps': True},
}

#: param_reduction_options of each statistic; the converged ones need jacobian_check_steps.
STATISTICS = {
    'rms': {'redundancy_statistic': 'rms'},
    'median': {'redundancy_statistic': 'median'},
    'rms_converged': {'redundancy_statistic': 'rms', 'redundancy_converged_only': True},
    'median_converged': {'redundancy_statistic': 'median', 'redundancy_converged_only': True},
}

STUDY_INPUTS = ('inputs', 'synthetic', 'generated_models')


def fisher_count(study, model):
    """Eigenvalues of at least 1 of the Fisher information at the truth (make_page's
    identifiable_count; make_page itself reads sys.argv when imported), and the parameters."""
    with open(os.path.join(study, model, 'compare', 'summary.json')) as f:
        summary = json.load(f)
    eigenvalues = summary['fisher_at_truth']['eigenvalues']
    return int(sum(e >= 1.0 for e in eigenvalues)), summary['parameters']


def reduce(sa_dir, params_for_id, out_dir, options):
    """The kept parameters and the report of a pca reduction of ``sa_dir`` (None when the
    variant cannot run it: no second step for a converged statistic)."""
    import warnings
    from libcuflynx.sensitivity_analysis.parameter_reduction import (
        PCASobolReduction, SensitivityFilesError)
    report = os.path.join(out_dir, 'param_reduction.json')
    if not os.path.isfile(report):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                PCASobolReduction(**options).reduce(sa_dir, params_for_id,
                                                    os.path.join(out_dir, 'kept.json'))
        except SensitivityFilesError as exc:
            return {'skipped': str(exc)}
    with open(report) as f:
        result = json.load(f)
    return {'kept': [row['parameter'] for row in result['selected']],
            'n_kept': len(result['selected']),
            'unexplained': result.get('unexplained_effects', {}),
            'points_used': result.get('jacobian_points_used'),
            'unconverged_points': result.get('unconverged_points', [])}


def table(results):
    lines = ['| model | Fisher count | variant | ' + ' | '.join(STATISTICS) + ' |',
             '|---|---|---|' + '---|' * len(STATISTICS)]
    for model, entry in results.items():
        for variant, by_statistic in entry['variants'].items():
            cells = []
            for statistic in STATISTICS:
                r = by_statistic.get(statistic, {})
                cells.append('-' if 'n_kept' not in r else
                             f"{r['n_kept']}" + (f" ({len(r['unconverged_points'])} unconv.)"
                                                 if r['unconverged_points'] else ''))
            lines.append(f"| {model} | {entry['fisher_count']} | {variant} | "
                         + ' | '.join(cells) + ' |')
    return '\n'.join(lines) + '\n'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('study')
    parser.add_argument('out')
    parser.add_argument('--models', nargs='+', default=['fitzhugh_nagumo', 'goodwin', 'teusink'],
                        choices=sorted(SPECS))
    parser.add_argument('--num-samples', type=int, default=256)
    parser.add_argument('--variants', nargs='+', default=list(DERIVATIVE_VARIANTS),
                        choices=list(DERIVATIVE_VARIANTS))
    args = parser.parse_args(argv)
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    study, root = os.path.abspath(args.study), os.path.abspath(args.out)

    results = {}
    for model in args.models:
        spec = SPECS[model]
        count, names = fisher_count(study, model)
        results[model] = {'fisher_count': count, 'parameters': names, 'variants': {}}
        for variant in args.variants:
            out = os.path.join(root, model, variant)
            p = paths(out)
            if rank == 0:
                for name in STUDY_INPUTS:
                    if not os.path.exists(os.path.join(out, name)):
                        shutil.copytree(os.path.join(study, model, name), os.path.join(out, name))
            comm.Barrier()
            done = os.path.join(p['sa'], 'sobol_problem.json')
            if not comm.bcast(os.path.isfile(done) if rank == 0 else None, root=0):
                if rank == 0:
                    print(f'[{model} {variant}] Sobol run', flush=True)
                from libcuflynx.scripts.sensitivity_analysis_run_script import run_SA
                run_SA(raw_inputs(
                    spec, out, p['sa_obs'], p['params_for_id'], os.path.join(out, 'sa_run'),
                    sa_options={'method': 'sobol', 'sample_type': 'saltelli',
                                'num_samples': args.num_samples, 'output_dir': p['sa'],
                                **DERIVATIVE_VARIANTS[variant]},
                    do_param_reduction=False))
                comm.Barrier()
            if rank == 0:
                results[model]['variants'][variant] = {
                    statistic: reduce(p['sa'], p['params_for_id'],
                                      os.path.join(out, 'reductions', statistic), options)
                    for statistic, options in STATISTICS.items()}
    if rank == 0:
        with open(os.path.join(root, 'redundancy_comparison.json'), 'w') as f:
            json.dump({'derivative_variants': DERIVATIVE_VARIANTS, 'statistics': STATISTICS,
                       'results': results}, f, indent=1)
        with open(os.path.join(root, 'redundancy_comparison.md'), 'w') as f:
            f.write(table(results))
        print(table(results), flush=True)


if __name__ == '__main__':
    main()
