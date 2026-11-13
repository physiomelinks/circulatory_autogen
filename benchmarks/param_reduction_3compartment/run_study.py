"""The 3compartment parameter-reduction study: one Sobol screen of seven parameters, a "pca" and
a "sobol" (simple) reduction of it keeping the same number of parameters, and calibrations of
all seven and of each reduction, from several CMA-ES seeds (CMA-ES spreads each
generation over the MPI ranks and stops at its call budget).

Run after ``make_synthetic_obs.py <out>``, under MPI::

    mpiexec -n 4 python run_study.py <out> [--num-samples 256] [--calls 2000] [--seeds 0 1 2]

It writes ``<out>/calibration_workflow.json`` and runs it into ``<out>/run``, one step at a
time; a step whose result is already there is not run again, so a stopped study resumes.
``reduce_simple`` keeps as many parameters as ``reduce_pca`` did, so it is set after that step.
Then ``python compare.py <out>``.
"""

import argparse
import json
import os
import sys

from make_synthetic_obs import OBS_FILE, PARAMS_FOR_ID_FILE, PREFIX

WORKFLOW = 'calibration_workflow.json'
METHODS = ('full', 'pca', 'simple')


def workflow(num_samples, calls, seeds, n_simple):
    model = f'inputs/{PREFIX}_module_array.csv'
    data = {'model': model, 'obs_data': f'inputs/{OBS_FILE}'}

    def fit(method, seed):
        step = dict(data, id=f'fit_{method}_s{seed}',
                    description=f'CMA-ES calibration of the {method} parameter set, seed {seed}',
                    settings={'optimiser_options': {'seed': seed}})
        if method == 'full':
            step['params_for_id'] = f'inputs/{PARAMS_FOR_ID_FILE}'
        else:
            step['params_for_id_from'] = f'reduce_{method}'
        return step

    return {
        'schema_version': 1, 'workflow_name': 'param_reduction_3compartment',
        'description': ('3compartment with seven parameters against six synthetic features: '
                        'the full calibration, and calibrations of a PCA-mode reduction and of '
                        'a top-N Sobol reduction of the same size'),
        'settings': {
            'model_type': 'cellml', 'solver': 'CVODE_myokit',
            'solver_info': {'MaximumStep': 0.001}, 'pre_time': 20, 'sim_time': 2, 'dt': 0.01,
            'param_id_method': 'CMA-ES',
            'optimiser_options': {'num_calls_to_function': calls, 'max_patience': 60,
                                  'cost_convergence': 1e-10, 'cost_type': 'gaussian_MLE'}},
        'target': model,
        'steps': [
            dict(data, id='screen', kind='sensitivity',
                 params_for_id=f'inputs/{PARAMS_FOR_ID_FILE}',
                 description='Sobol screen of the seven parameters',
                 settings={'sa_options': {'method': 'sobol', 'sample_type': 'saltelli',
                                          'num_samples': num_samples}}),
            {'id': 'reduce_pca', 'kind': 'reduction', 'sensitivity_from': 'screen',
             'description': 'one parameter per informative PCA mode',
             'settings': {'param_reduction_options': {'method': 'pca'}}},
            {'id': 'reduce_simple', 'kind': 'reduction', 'sensitivity_from': 'screen',
             'description': 'top-N by max ST over the outputs, N as reduce_pca kept',
             'settings': {'param_reduction_options': {'method': 'sobol',
                                                      'n_params': n_simple}}},
            *[fit(method, seed) for method in METHODS for seed in seeds],
        ]}


def main(argv=None):
    from libcuflynx.calibration_workflow import run_calibration_workflow
    from libcuflynx.utilities.mpi_utils import get_MPI
    comm = get_MPI().COMM_WORLD
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('out')
    parser.add_argument('--num-samples', type=int, default=256)
    parser.add_argument('--calls', type=int, default=2000)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    args = parser.parse_args(argv)
    out = os.path.abspath(args.out)
    run_dir = os.path.join(out, 'run')
    path = os.path.join(out, WORKFLOW)

    def result(step_id):
        path = os.path.join(run_dir, step_id, 'step_result.json')
        if not os.path.isfile(path):
            return None
        with open(path) as f:
            return json.load(f)

    def write(n_simple):
        if comm.Get_rank() == 0:
            with open(path, 'w') as f:
                json.dump(workflow(args.num_samples, args.calls, args.seeds, n_simple), f,
                          indent=1)
        comm.Barrier()

    kept = result('reduce_pca')
    write(kept['kept_rows'] if kept else 1)
    for step in workflow(args.num_samples, args.calls, args.seeds, 1)['steps']:
        if result(step['id']) is None:
            if comm.Get_rank() == 0:
                print(f'[study] running {step["id"]}', flush=True)
            run_calibration_workflow(path, output_dir=run_dir, only=step['id'], comm=comm)
            comm.Barrier()
        if step['id'] == 'reduce_pca':
            write(result('reduce_pca')['kept_rows'])
    if comm.Get_rank() == 0:
        print(f'[study] done: {run_dir}', flush=True)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception:
        # one rank's failure would leave the others waiting in a barrier
        import traceback
        traceback.print_exc()
        from libcuflynx.utilities.mpi_utils import get_MPI
        get_MPI().COMM_WORLD.Abort(1)
