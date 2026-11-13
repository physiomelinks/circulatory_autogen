"""Run one model of the parameter-reduction study, after ``make_inputs.py``, under MPI::

    mpiexec -n 4 python run_study.py <model> <out> [--num-samples 256] [--calls 3000]
        [--seeds 0 1 2] [--methods full pca simple] [--stage all|sa]

1. Sobol analysis of all the parameters against the binned obs_data, followed by the PCA
   reduction (``do_param_reduction``, ``param_reduction_options: {method: pca}``), through
   ``run_SA`` as ``run_sensitivity_analysis.sh`` runs it.
2. The simple reduction: top-n by max ST, n the number the PCA reduction kept.
3. CMA-ES calibrations against the series of all the parameters (``full``) and of each
   reduction, one per seed, through ``run_param_id``. The parameters a reduction drops keep
   the model's defaults.

Each stage writes its result into ``<out>`` and is skipped when that result is there, so a
stopped study resumes.
"""

import argparse
import json
import os
import sys

from models import SPECS, paths, raw_inputs

METHODS = ('full', 'pca', 'simple')
REDUCED = 'reduced_params_for_id.json'


def main(argv=None):
    from libcuflynx.utilities.mpi_utils import get_MPI
    comm = get_MPI().COMM_WORLD
    rank = comm.Get_rank()
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('model', choices=sorted(SPECS))
    parser.add_argument('out')
    parser.add_argument('--num-samples', type=int, default=256)
    parser.add_argument('--calls', type=int, default=3000)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--methods', nargs='+', choices=METHODS, default=list(METHODS))
    parser.add_argument('--stage', choices=('all', 'sa'), default='all')
    args = parser.parse_args(argv)
    spec = SPECS[args.model]
    out = os.path.abspath(args.out)
    p = paths(out)
    reduced = {m: os.path.join(p['reductions'], m, REDUCED) for m in ('pca', 'simple')}

    def say(text):
        if rank == 0:
            print(f'[study {args.model}] {text}', flush=True)

    if not os.path.isfile(reduced['pca']):
        say('Sobol analysis and PCA reduction')
        from libcuflynx.scripts.sensitivity_analysis_run_script import run_SA
        run_SA(raw_inputs(
            spec, out, p['sa_obs'], p['params_for_id'], os.path.join(out, 'sa_run'),
            sa_options={'method': 'sobol', 'sample_type': 'saltelli',
                        'num_samples': args.num_samples, 'output_dir': p['sa']},
            do_param_reduction=True,
            param_reduction_options={'method': 'pca',
                                     'output_dir': os.path.dirname(reduced['pca'])}))
        comm.Barrier()
    if args.stage == 'sa':
        return

    if not os.path.isfile(reduced['simple']) and rank == 0:
        from libcuflynx.sensitivity_analysis.parameter_reduction import ParamReduction
        with open(os.path.join(os.path.dirname(reduced['pca']), 'param_reduction.json')) as f:
            n = len(json.load(f)['selected'])
        say(f'simple reduction, n = {n}')
        ParamReduction.from_options({'method': 'sobol', 'n_params': n}).reduce(
            p['sa'], p['params_for_id'], reduced['simple'])
    comm.Barrier()

    from libcuflynx.scripts.param_id_run_script import run_param_id
    for method in args.methods:
        params_for_id = p['params_for_id'] if method == 'full' else reduced[method]
        with open(params_for_id) as f:
            targets = [entry['targets'][0] for entry in json.load(f)['params']]
        for seed in args.seeds:
            fit_dir = os.path.join(p['fits'], f'{method}_s{seed}')
            result = os.path.join(fit_dir, 'result.json')
            if os.path.isfile(result):
                continue
            say(f'calibrating {method}, seed {seed}: {targets}')
            run = run_param_id(raw_inputs(
                spec, out, p['obs'], params_for_id, fit_dir, param_id_method='CMA-ES',
                optimiser_options={'num_calls_to_function': args.calls, 'seed': seed,
                                   'max_patience': 60, 'cost_convergence': 1e-10,
                                   'cost_type': 'gaussian_MLE'}))
            if rank == 0:
                with open(result, 'w') as f:
                    json.dump({'method': method, 'seed': seed, 'calibrated': targets,
                               'best': [float(v) for v in run['best_param_vals']],
                               'params_for_id': params_for_id,
                               'param_id_output_dir': run['output_dir']}, f, indent=1)
            comm.Barrier()
    say('done')


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception:
        import traceback
        traceback.print_exc()
        from libcuflynx.utilities.mpi_utils import get_MPI
        get_MPI().COMM_WORLD.Abort(1)
