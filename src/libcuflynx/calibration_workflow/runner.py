'''
Running a calibration workflow: every step through the normal param_id path, in dependency
order, then merging the results into the workflow's target.

Output layout (``output_dir``)::

    calibration_workflow.json      the workflow as run
    workflow_result.json           order, per-step provenance, the merged parameter set
    <step_id>/
        step_result.json           calibrated values, fixed values and priors used, input hashes
        model/                     the generated step model (resources/, generated_models/)
        param_id/                  the normal param_id outputs (best_param_vals.npy, ...)
        <step_id>_params_for_id.csv   only when priors_from adds parameters to the step's own
        distribution/              only with store_distribution (see distributions.py)
    _target/model/                 the target generated with every calibrated value

Under MPI every rank runs every step (calibration is parallel); generation and every file
write happen on rank 0, and what rank 0 reads is broadcast.
'''

import copy
import datetime
import hashlib
import json
import os
import shutil
import time

import numpy as np

from libcuflynx.calibration_workflow import naming
from libcuflynx.calibration_workflow.distributions import (DISTRIBUTION_DIR,
                                                           StoredDistribution,
                                                           parameters_from_info)
from libcuflynx.calibration_workflow.generate import generate_module_instance, read_parameter_rows
from libcuflynx.calibration_workflow.resolve import resolve_workflow
from libcuflynx.calibration_workflow.spec import (WORKFLOW_FILE_NAME, WorkflowError, load_workflow,
                                                  uses_log_posterior)

STEP_RESULT = 'step_result.json'
# where a step's calibrated values come from
POINT_BEST_FIT = 'best_fit'
POINT_POSTERIOR_MEDIAN = 'posterior_median'
WORKFLOW_RESULT = 'workflow_result.json'
TARGET_DIR = '_target'

# what a step's user_inputs get unless the workflow's settings say otherwise
STEP_DEFAULTS = {'model_type': 'cellml', 'DEBUG': False, 'do_ad': False, 'do_ia': False,
                 'UQ_options': {}}


class WorkflowStepError(RuntimeError):
    '''A step that failed while running.'''

    def __init__(self, step_id, message):
        super().__init__(f'step "{step_id}": {message}')
        self.step_id = step_id


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def _now():
    return datetime.datetime.now().isoformat(timespec='seconds')


def _get_comm(comm):
    if comm is not None:
        return comm
    from libcuflynx.utilities.mpi_utils import get_MPI
    return get_MPI().COMM_WORLD


def _libcuflynx_version():
    try:
        from importlib.metadata import version
        return version('libcuflynx')
    except Exception:
        return None


def default_output_dir(workflow):
    return os.path.join(os.getcwd(), 'workflow_output', workflow.name)


def _write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def _read_json(path):
    with open(path) as f:
        return json.load(f)


class _Run:
    '''One run of a resolved workflow.'''

    def __init__(self, resolved, output_dir, comm, progress):
        self.resolved = resolved
        self.workflow = resolved.workflow
        self.output_dir = os.path.abspath(output_dir)
        self.comm = comm
        self.rank = comm.Get_rank()
        self.progress = progress

    # -- helpers -------------------------------------------------------------------------

    def emit(self, event, **fields):
        if self.progress is not None and self.rank == 0:
            self.progress(dict(event=event, workflow=self.workflow.name, **fields))

    def on_root(self, fn):
        '''``fn()`` on rank 0, its result (or the exception it raised) on every rank.'''
        outcome = None
        if self.rank == 0:
            try:
                outcome = ('ok', fn())
            except BaseException as exc:  # noqa: B902 - re-raised on every rank below
                outcome = ('error', exc)
        outcome = self.comm.bcast(outcome, root=0)
        if outcome[0] == 'error':
            raise outcome[1]
        return outcome[1]

    def step_dir(self, step_id):
        return os.path.join(self.output_dir, step_id)

    def step_result(self, step_id):
        path = os.path.join(self.step_dir(step_id), STEP_RESULT)
        if not os.path.isfile(path):
            return None
        return _read_json(path)

    def inputs(self, step_id):
        instance = self.resolved.steps[step_id].instance
        files = {'obs_data': instance.obs_data_path,
                 'params_for_id': instance.params_for_id_path}
        if os.path.isfile(instance.parameters_path):
            files['parameters'] = instance.parameters_path
        return {key: {'path': path, 'sha256': sha256(path)} for key, path in files.items()}

    def require_current(self, step_id, needed_by):
        result = self.step_result(step_id)
        if result is None or result.get('status') != 'done':
            raise WorkflowError(
                f'step "{needed_by}" needs the result of "{step_id}", which has not been run '
                f'into {self.output_dir}. Run the workflow from "{step_id}" (or without '
                f'--from-step/--only).')
        current = self.inputs(step_id)
        for key, info in result.get('inputs', {}).items():
            if key in current and current[key]['sha256'] != info.get('sha256'):
                raise WorkflowError(
                    f'step "{needed_by}" needs the result of "{step_id}", but '
                    f'{current[key]["path"]} has changed since "{step_id}" ran. Run the '
                    f'workflow from "{step_id}".')
        return result

    # -- what a step gets from earlier steps ---------------------------------------------

    def fixed_values(self, step):
        '''``{model name in this step: row}`` from every fixed_from step: its calibrated
        values and the values it was itself given as fixed.'''
        values = {}
        for source in step.fixed_from:
            result = self.require_current(source, step.id)
            path = self.resolved.path_between(source, step.id)
            for record in result['calibrated'] + result['fixed']:
                name = naming.model_name(naming.map_vessel(record['vessel'], path),
                                         record['param'])
                origin = record.get('from_step', source)
                if name in values and not np.isclose(values[name]['value'], record['value'],
                                                      rtol=1e-12, atol=0.0):
                    raise WorkflowError(
                        f'step "{step.id}" is given two different fixed values for "{name}": '
                        f'{values[name]["value"]} (from "{values[name]["from_step"]}") and '
                        f'{record["value"]} (from "{origin}").')
                values[name] = {'vessel': naming.map_vessel(record['vessel'], path),
                                'param': record['param'], 'model_name': name,
                                'value': record['value'], 'units': record.get('units', ''),
                                'from_step': origin}
        return values

    def priors(self, step):
        '''``[(PriorSource, StoredDistribution named in this step)]``.'''
        out = []
        for prior in step.priors_from:
            self.require_current(prior.step, step.id)
            distribution = StoredDistribution.load(
                os.path.join(self.step_dir(prior.step), DISTRIBUTION_DIR))
            if prior.parameters is not None:
                distribution = distribution.subset(list(prior.parameters))
            path = self.resolved.path_between(prior.step, step.id)
            out.append((prior, distribution.mapped_into(path)))
        return out

    # -- one step --------------------------------------------------------------------------

    def run_step(self, step, index, total):
        from libcuflynx.scripts.param_id_run_script import run_param_id

        resolved = self.resolved.steps[step.id]
        step_dir = self.step_dir(step.id)
        started, t0 = _now(), time.time()
        self.emit('step_started', step=step.id, index=index, total=total,
                  target=step.target.as_dict())

        settings = self.workflow.step_settings(step)

        def prepare():
            own = {p.model_name for p in resolved.parameters}
            fixed = self.fixed_values(step)
            clash = sorted(own & set(fixed))
            if clash:
                raise WorkflowError(
                    f'step "{step.id}" calibrates {clash}, which it is also given as fixed '
                    f'values by {sorted({fixed[n]["from_step"] for n in clash})}. A parameter '
                    f'is either fixed or calibrated; remove it from one of them.')
            priors = self.priors(step)
            prior_names = [n for _, dist in priors for n in dist.names]
            clash = sorted((own | set(fixed)) & set(prior_names))
            if clash:
                raise WorkflowError(f'step "{step.id}" is given priors for {clash}, which it '
                                    f'also calibrates or is given as fixed values.')
            if os.path.isdir(step_dir):
                shutil.rmtree(step_dir)
            os.makedirs(step_dir)
            overrides = [{'variable_name': name, 'units': row['units'],
                          'value': repr(float(row['value'])),
                          'data_reference': f'fixed by calibration workflow '
                                            f'{self.workflow.name}, step {row["from_step"]}'}
                         for name, row in sorted(fixed.items())]
            generated = generate_module_instance(
                step.target, os.path.join(step_dir, 'model'), step.id, overrides,
                library_inputs=self.resolved.library_inputs,
                model_type=settings.get('model_type', STEP_DEFAULTS['model_type']),
                solver=settings.get('solver'))
            model_rows = {r['variable_name']: r
                          for r in read_parameter_rows(generated['parameters_path'])}
            for name, row in fixed.items():
                if name not in model_rows:
                    raise WorkflowError(
                        f'step "{step.id}": the value fixed from "{row["from_step"]}" for '
                        f'"{name}" has no parameter of that name in the step model '
                        f'({generated["parameters_path"]}).')
            params_for_id = resolved.instance.params_for_id_path
            if priors:
                params_for_id = os.path.join(step_dir, f'{step.id}_params_for_id.csv')
                _write_params_for_id_with_priors(resolved.instance.params_for_id_path,
                                                 params_for_id, priors, step.id)
            units = {k: v.get('units', '') for k, v in model_rows.items()}
            return generated, params_for_id, units, fixed, priors

        generated, params_for_id, units, fixed, priors = self.on_root(prepare)

        inp = copy.deepcopy(STEP_DEFAULTS)
        inp.update(copy.deepcopy(settings))
        inp.update({
            'file_prefix': step.id,
            'resources_dir': generated['resources_dir'],
            'generated_models_dir': generated['generated_models_dir'],
            'input_param_file': f'{step.id}_parameters.csv',
            'param_id_obs_path': resolved.instance.obs_data_path,
            'params_for_id_file': params_for_id,
            'param_id_output_dir': os.path.join(step_dir, 'param_id'),
            'do_uq': step.store_distribution,
        })
        inp.update({k: v for k, v in self.resolved.library_inputs.items()})
        if priors:
            inp['joint_priors'] = [([n for n in dist.names],
                                    dist.as_prior(prior.kind, prior.inflate))
                                   for prior, dist in priors]
        try:
            outcome = run_param_id(inp)
        except SystemExit as exc:
            # the user_inputs parser exit()s on a bad configuration, after printing why
            raise WorkflowStepError(step.id, f'the param_id configuration was refused (exit '
                                             f'{exc.code}); the messages above say why.')

        def finish():
            case_dir = outcome['output_dir']
            from libcuflynx.parsers.PrimitiveParsers import ObsAndParamDataParser
            info = ObsAndParamDataParser().get_param_id_info(params_for_id)
            best = np.atleast_1d(np.load(os.path.join(case_dir, 'best_param_vals.npy')))
            best_cost = float(np.load(os.path.join(case_dir, 'best_cost.npy')))
            distribution_dir = None
            if step.store_distribution:
                distribution_dir = self._store_distribution(step, case_dir, info, settings)
            values, point_estimate = best, POINT_BEST_FIT
            if priors and not uses_log_posterior(settings):
                # The optimiser minimised the cost, which knows nothing of the priors; only the
                # MCMC sampled the posterior they shape. Report that posterior's medians.
                samples = StoredDistribution.load(distribution_dir).samples
                values, point_estimate = np.median(samples, axis=0), POINT_POSTERIOR_MEDIAN
            calibrated = []
            for qnames, value in zip(info['param_names'], values):
                for qname in (qnames if isinstance(qnames, (list, tuple)) else [qnames]):
                    vessel, _, param = str(qname).partition('/')
                    name = naming.model_name(vessel, param)
                    target_vessel = naming.map_vessel(vessel, resolved.path)
                    calibrated.append({
                        'vessel': vessel, 'param': param, 'model_name': name,
                        'target_model_name': naming.model_name(target_vessel, param),
                        'target_instance_name': naming.instance_name(target_vessel, param),
                        'value': float(value), 'units': units.get(name, '')})
            result = {
                'step': step.id, 'status': 'done', 'target': step.target.as_dict(),
                'submodule_path': resolved.path, 'started': started, 'finished': _now(),
                'duration_s': round(time.time() - t0, 3),
                'method': settings.get('param_id_method'), 'best_cost': best_cost,
                'point_estimate': point_estimate,
                'best_fit': [float(v) for v in best],
                'calibrated': calibrated,
                'fixed': [dict(row) for _, row in sorted(fixed.items())],
                'priors': [dict(prior.as_dict(), parameters=dist.names)
                           for prior, dist in priors],
                'inputs': self.inputs(step.id),
                'params_for_id_used': params_for_id,
                'settings': settings,
                'model_path': generated['model_path'],
                'param_id_output_dir': case_dir,
                'distribution_dir': distribution_dir,
            }
            _write_json(os.path.join(step_dir, STEP_RESULT), result)
            return result

        result = self.on_root(finish)
        self.emit('step_finished', step=step.id, index=index, total=total,
                  best_cost=result['best_cost'], duration_s=result['duration_s'])
        return result

    def _store_distribution(self, step, case_dir, info, settings):
        chain_path = os.path.join(case_dir, 'mcmc_chain.npy')
        if not os.path.isfile(chain_path):
            raise WorkflowStepError(step.id, f'store_distribution is set but MCMC wrote no '
                                             f'chain ({chain_path}).')
        chain = np.load(chain_path)
        uq = settings.get('debug_UQ_options') if settings.get('DEBUG') else None
        uq = uq or settings.get('UQ_options') or {}
        burn_in = float(uq.get('burn_in', 0.5))
        start = int(chain.shape[0] * burn_in) if burn_in < 1 else int(burn_in)
        parameters = parameters_from_info(info['param_names'], info['param_mins'],
                                          info['param_maxs'])
        distribution = StoredDistribution.from_chain(
            chain, parameters, burn_in_index=start,
            source={'workflow': self.workflow.name, 'step': step.id,
                    'target': step.target.as_dict(),
                    'obs_data_sha256': sha256(self.resolved.steps[step.id]
                                              .instance.obs_data_path),
                    'chain': chain_path, 'created': _now()})
        return distribution.save(os.path.join(self.step_dir(step.id), DISTRIBUTION_DIR),
                                 chain=chain)

    # -- the merged result -----------------------------------------------------------------

    def merge(self, order, generate_target=True, write_calibrated=False):
        results = {s.id: self.step_result(s.id) for s in self.workflow.steps}
        merged = {}
        for step in order:
            result = results.get(step.id)
            if not result or result.get('status') != 'done':
                continue
            for record in result['calibrated']:
                merged[record['target_model_name']] = {
                    'model_name': record['target_model_name'],
                    'instance_name': record['target_instance_name'],
                    'value': record['value'], 'units': record.get('units', ''),
                    'from_step': step.id}
        complete = all(r and r.get('status') == 'done' for r in results.values())
        target = self.resolved.target
        workflow_result = {
            'workflow_name': self.workflow.name,
            'workflow_file': self.workflow.path,
            'workflow_sha256': sha256(self.workflow.path) if self.workflow.path else None,
            'target': self.workflow.target.as_dict(),
            'order': [s.id for s in order],
            'complete': complete,
            'finished': _now(),
            'libcuflynx_version': _libcuflynx_version(),
            'module_library_dirs': self.workflow.module_library_dirs,
            'steps': {sid: (None if r is None else {
                'status': r.get('status'), 'target': r.get('target'),
                'submodule_path': r.get('submodule_path'), 'started': r.get('started'),
                'finished': r.get('finished'), 'best_cost': r.get('best_cost'),
                'inputs': r.get('inputs'), 'result': os.path.join(sid, STEP_RESULT)})
                for sid, r in results.items()},
            'merged_parameters': sorted(merged.values(), key=lambda r: r['model_name']),
            'target_model': None,
            'calibrated_parameters_file': None,
        }
        if generate_target and merged:
            overrides = [{'variable_name': r['model_name'], 'units': r['units'],
                          'value': repr(float(r['value'])),
                          'data_reference': f'calibration workflow {self.workflow.name}, '
                                            f'step {r["from_step"]}'}
                         for r in workflow_result['merged_parameters']]
            target_dir = os.path.join(self.output_dir, TARGET_DIR)
            if os.path.isdir(target_dir):
                shutil.rmtree(target_dir)
            generated = generate_module_instance(
                self.workflow.target, os.path.join(target_dir, 'model'), 'target', overrides,
                library_inputs=self.resolved.library_inputs,
                model_type=self.workflow.settings.get('model_type', 'cellml'),
                solver=self.workflow.settings.get('solver'))
            workflow_result['target_model'] = {
                'model_path': generated['model_path'],
                'parameters_path': generated['parameters_path']}
        if write_calibrated and merged:
            if not complete:
                raise WorkflowError('the calibrated parameters file is written only for a '
                                    'complete workflow run; some steps have no result.')
            workflow_result['calibrated_parameters_file'] = write_calibrated_parameters(
                target, workflow_result['merged_parameters'], self.workflow.name)
        _write_json(os.path.join(self.output_dir, WORKFLOW_RESULT), workflow_result)
        return workflow_result


def write_calibrated_parameters(target, merged_parameters, workflow_name):
    '''``<instance>_calibrated_parameters.csv`` in the target instance's directory: its
    parameters with the merged values in place (rows the instance file lacks are added).'''
    rows = read_parameter_rows(target.parameters_path)
    columns = list(rows[0].keys()) if rows else ['variable_name', 'units', 'value',
                                                  'data_reference']
    by_name = {r['variable_name']: r for r in rows}
    today = datetime.date.today().isoformat()
    for record in merged_parameters:
        reference = (f'calibrated by workflow {workflow_name}, step {record["from_step"]} '
                     f'({today})')
        row = by_name.get(record['instance_name'])
        if row is None:
            row = {c: '' for c in columns}
            row.update(variable_name=record['instance_name'], units=record['units'])
            rows.append(row)
            by_name[record['instance_name']] = row
        row['value'] = '%.10g' % record['value']
        row['data_reference'] = reference
        if 'sourced' in columns:
            row['sourced'] = 'no'
    path = target.file('calibrated_parameters.csv')
    import csv
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_params_for_id_with_priors(source_path, out_path, priors, step_id):
    '''The step's own params_for_id plus one ``uniform`` row (for the bounds) per parameter a
    prior is given for; the prior's density itself is added as a joint prior.'''
    import csv
    with open(source_path, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        columns = list(reader.fieldnames)
        rows = [dict(r) for r in reader]
    for column in ('vessel_name', 'param_name', 'min', 'max', 'name_for_plotting'):
        if column not in columns:
            columns.append(column)
    for prior, distribution in priors:
        for par in distribution.parameters:
            params = {p for _, p in par.targets}
            if len(params) != 1:
                raise WorkflowError(f'step "{step_id}": a prior from "{prior.step}" is over a '
                                    f'grouped parameter with targets {par.targets} of '
                                    f'different names, which params_for_id cannot express.')
            if not (np.isfinite(par.min) and np.isfinite(par.max)):
                raise WorkflowError(f'step "{step_id}": the prior from "{prior.step}" over '
                                    f'{par.model_names()} has no finite bounds.')
            row = {c: '' for c in columns}
            row.update(vessel_name=' '.join(v for v, _ in par.targets), param_name=params.pop(),
                       min=repr(par.min), max=repr(par.max),
                       name_for_plotting=par.model_names()[0])
            if 'prior' in columns:
                row['prior'] = 'uniform'
            rows.append(row)
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


# --------------------------------------------------------------------------------------------
# the public entry points
# --------------------------------------------------------------------------------------------

def plan_workflow(source, module_library_dirs=None):
    '''What running the workflow would do, without running anything: the order, each step's
    files, where its values land in the target, and what it is given by earlier steps.'''
    workflow = source if hasattr(source, 'steps') else load_workflow(source, module_library_dirs)
    resolved = resolve_workflow(workflow)
    steps = []
    for step in workflow.topological_order():
        rs = resolved.steps[step.id]
        fixed = []
        for source_id in step.fixed_from:
            path = resolved.path_between(source_id, step.id)
            for p in resolved.steps[source_id].parameters:
                fixed.append({'from_step': source_id, 'model_name': p.model_name,
                              'in_step': naming.model_name(naming.map_vessel(p.vessel, path),
                                                           p.param)})
        steps.append({
            'id': step.id, 'target': step.target.as_dict(),
            'submodule_path': rs.path, 'depends_on': list(step.depends_on),
            'obs_data': rs.instance.obs_data_path,
            'params_for_id': rs.instance.params_for_id_path,
            'calibrates': [{'model_name': p.model_name,
                            'target_model_name': rs.target_model_name(p),
                            'target_instance_name': rs.target_instance_name(p)}
                           for p in rs.parameters],
            'fixed_from': fixed,
            'priors_from': [p.as_dict() for p in step.priors_from],
            'store_distribution': step.store_distribution,
            'settings': workflow.step_settings(step),
        })
    return {'workflow_name': workflow.name, 'target': workflow.target.as_dict(),
            'target_parameters': resolved.target.parameters_path,
            'module_library_dirs': workflow.module_library_dirs, 'steps': steps}


def run_calibration_workflow(source, *, output_dir=None, module_library_dirs=None,
                             from_step=None, only=None, dry_run=False, write_calibrated=False,
                             progress=None, comm=None):
    '''Run the calibration workflow ``source`` (a calibration_workflow.json path or its dict).

    Args:
        output_dir: where the run goes (default ``./workflow_output/<workflow_name>``).
        module_library_dirs: module libraries, added to the file's own.
        from_step: run this step and every step after it in the order; earlier steps' results
            are reused from ``output_dir`` (an error if missing, or if their obs_data,
            params_for_id or parameters changed since).
        only: run just this step, reusing the results it depends on.
        dry_run: resolve and check everything, run nothing; returns :func:`plan_workflow`.
        write_calibrated: write the merged set as the target instance's
            ``<instance>_calibrated_parameters.csv`` (complete runs only).
        progress: ``callable(event_dict)``, called on rank 0 with ``event`` in
            ``workflow_started``, ``step_started``, ``step_finished``, ``step_failed``,
            ``step_skipped``, ``workflow_finished``.
        comm: an MPI communicator (default COMM_WORLD, or a one-rank stand-in).

    Returns the workflow_result dict (also written as ``workflow_result.json``).
    '''
    workflow = load_workflow(source, module_library_dirs)
    if from_step and only:
        raise WorkflowError('give from_step or only, not both.')
    for name in (from_step, only):
        if name:
            workflow.step(name)  # raises for an unknown id
    if dry_run:
        return plan_workflow(workflow)

    comm = _get_comm(comm)
    resolved = resolve_workflow(workflow)
    output_dir = os.path.abspath(output_dir or default_output_dir(workflow))
    run = _Run(resolved, output_dir, comm, progress)
    order = workflow.topological_order()
    if only:
        to_run = [workflow.step(only)]
    elif from_step:
        ids = [s.id for s in order]
        to_run = order[ids.index(from_step):]
    else:
        to_run = order

    def start():
        os.makedirs(output_dir, exist_ok=True)
        snapshot = os.path.join(output_dir, WORKFLOW_FILE_NAME)
        if workflow.path:
            shutil.copyfile(workflow.path, snapshot)
        else:
            _write_json(snapshot, workflow.raw)
    run.on_root(start)
    run.emit('workflow_started', order=[s.id for s in order], running=[s.id for s in to_run],
             output_dir=output_dir)
    for step in order:
        if step not in to_run:
            run.emit('step_skipped', step=step.id)
    for index, step in enumerate(to_run):
        try:
            run.run_step(step, index, len(to_run))
        except Exception as exc:
            run.emit('step_failed', step=step.id, index=index, total=len(to_run),
                     error=str(exc))
            raise
    result = run.on_root(lambda: run.merge(order, write_calibrated=write_calibrated))
    run.emit('workflow_finished', complete=result['complete'], output_dir=output_dir)
    return result


def workflow_status(source, output_dir, module_library_dirs=None):
    """Where each step of a workflow stands in ``output_dir``, cheaply (no generation).

    Returns ``{'workflow_name', 'order', 'complete', 'steps': {id: {'status', 'stale',
    'best_cost', 'finished', 'point_estimate'}}}`` with ``status`` ``'done'`` or
    ``'not_run'``; ``stale`` is True when the step's obs_data, params_for_id or parameters
    changed since it ran, or a step it takes values from is stale or was rerun after it.
    """
    workflow = source if hasattr(source, 'steps') else load_workflow(source, module_library_dirs)
    resolved = resolve_workflow(workflow)
    run = _Run(resolved, output_dir, _get_comm(None), None)
    steps = {}
    for step in workflow.topological_order():
        result = run.step_result(step.id)
        done = bool(result) and result.get('status') == 'done'
        stale = False
        if done:
            current = run.inputs(step.id)
            stale = any(key in current and current[key]['sha256'] != info.get('sha256')
                        for key, info in result.get('inputs', {}).items())
            for source_id in step.fixed_from + [p.step for p in step.priors_from]:
                upstream = steps.get(source_id, {})
                if upstream.get('stale') or upstream.get('status') != 'done' or \
                        (upstream.get('finished') or '') > (result.get('started') or ''):
                    stale = True
        steps[step.id] = {'status': 'done' if done else 'not_run', 'stale': stale,
                          'best_cost': (result or {}).get('best_cost') if done else None,
                          'finished': (result or {}).get('finished') if done else None,
                          'point_estimate': (result or {}).get('point_estimate') if done else None}
    return {'workflow_name': workflow.name, 'order': [s.id for s in workflow.topological_order()],
            'complete': all(v['status'] == 'done' and not v['stale'] for v in steps.values()),
            'steps': steps}


TARGET_VIEW = 'target'


def workflow_model(source, view, *, output_dir, work_dir, module_library_dirs=None,
                   with_own_results=True):
    """Generate the model one tab of a workflow shows, from whatever results exist so far.

    ``view`` is a step id, or ``'target'`` for the workflow's target. A step's model gets
    the values its ``fixed_from`` steps calibrated (those that have run into ``output_dir``)
    and, with ``with_own_results``, its own calibrated values once it has run. The target
    gets every calibrated value there is, mapped into its naming. Nothing is calibrated and
    nothing in ``output_dir`` is written: the model goes under ``work_dir``.

    Returns ``{view, kind ('step' or 'target'), target, submodule_path, model_path,
    flat_model_path (one self-contained CellML file), parameters_path, obs_data_path, params_for_id_path, fixed, calibrated, waiting_for,
    stale, result, param_id_output_dir}``. ``fixed`` lists the values given by earlier
    steps; ``calibrated`` the view's own (for the target, the merged set); ``waiting_for``
    the steps whose results it should have but that have not run; ``stale`` those whose
    inputs changed since they ran. obs_data / params_for_id are None when the instance has
    none (a target with no step of its own).
    """
    workflow = source if hasattr(source, 'steps') else load_workflow(source, module_library_dirs)
    resolved = resolve_workflow(workflow)
    run = _Run(resolved, output_dir, _get_comm(None), None)
    waiting, stale = [], []

    def available(step_id):
        result = run.step_result(step_id)
        if result is None or result.get('status') != 'done':
            waiting.append(step_id)
            return None
        current = run.inputs(step_id)
        if any(key in current and current[key]['sha256'] != info.get('sha256')
               for key, info in result.get('inputs', {}).items()):
            stale.append(step_id)
        return result

    overrides, fixed, calibrated = {}, [], []
    if view == TARGET_VIEW:
        kind, target = 'target', workflow.target
        for step in workflow.topological_order():
            result = available(step.id)
            for record in (result or {}).get('calibrated', []):
                overrides[record['target_model_name']] = (record, step.id)
        calibrated = [{'model_name': name, 'value': rec['value'], 'units': rec.get('units', ''),
                       'from_step': sid} for name, (rec, sid) in sorted(overrides.items())]
        own_step = next((s for s in workflow.steps if s.target == workflow.target), None)
        instance = resolved.target
        submodule_path = ''
        own_result = run.step_result(own_step.id) if own_step else None
    else:
        step = workflow.step(view)
        kind, target = 'step', step.target
        rs = resolved.steps[step.id]
        instance, submodule_path = rs.instance, rs.path
        for source_id in step.fixed_from:
            result = available(source_id)
            if result is None:
                continue
            path = resolved.path_between(source_id, step.id)
            for record in result['calibrated'] + result['fixed']:
                vessel = naming.map_vessel(record['vessel'], path)
                name = naming.model_name(vessel, record['param'])
                origin = record.get('from_step', source_id)
                overrides[name] = ({'value': record['value'], 'units': record.get('units', '')},
                                   origin)
                fixed.append({'model_name': name, 'value': record['value'],
                              'units': record.get('units', ''), 'from_step': origin})
        own_step = step
        own_result = run.step_result(step.id)
        if own_result and own_result.get('status') == 'done':
            calibrated = [{'model_name': r['model_name'], 'value': r['value'],
                           'units': r.get('units', ''), 'from_step': step.id}
                          for r in own_result['calibrated']]
            if with_own_results:
                for record in calibrated:
                    overrides.setdefault(record['model_name'], (record, step.id))
    rows = [{'variable_name': name, 'units': rec.get('units', ''),
             'value': repr(float(rec['value'])),
             'data_reference': f'calibration workflow {workflow.name}, step {sid}'}
            for name, (rec, sid) in sorted(overrides.items())]
    view_dir = os.path.join(os.path.abspath(work_dir), view)
    if os.path.isdir(view_dir):
        shutil.rmtree(view_dir)
    settings = workflow.step_settings(own_step) if kind == 'step' else workflow.settings
    generated = generate_module_instance(
        target, view_dir, view, rows, library_inputs=resolved.library_inputs,
        model_type=settings.get('model_type', STEP_DEFAULTS['model_type']),
        solver=settings.get('solver'))

    def existing(path):
        return path if path and os.path.isfile(path) else None

    has_own = own_step is not None
    return {
        'view': view, 'kind': kind, 'target': target.as_dict(),
        'step_id': own_step.id if has_own else None,
        'submodule_path': submodule_path,
        'model_path': generated['model_path'],
        'flat_model_path': generated['flat_model_path'],
        'parameters_path': generated['parameters_path'],
        'obs_data_path': existing(instance.obs_data_path) if has_own else None,
        'params_for_id_path': existing(instance.params_for_id_path) if has_own else None,
        'fixed': fixed, 'calibrated': calibrated,
        'waiting_for': sorted(set(waiting)), 'stale': sorted(set(stale)),
        'result': own_result,
        'param_id_output_dir': (own_result or {}).get('param_id_output_dir'),
    }


def load_workflow_run(output_dir):
    '''A workflow run read back from ``output_dir``: ``{'workflow': dict, 'result': dict or
    None, 'steps': {step id: step_result or None}}`` -- a partial run has some steps None.'''
    output_dir = os.path.abspath(output_dir)
    snapshot = os.path.join(output_dir, WORKFLOW_FILE_NAME)
    if not os.path.isfile(snapshot):
        raise WorkflowError(f'{output_dir} is not a calibration workflow run (no '
                            f'{WORKFLOW_FILE_NAME}).')
    raw = _read_json(snapshot)
    result_path = os.path.join(output_dir, WORKFLOW_RESULT)
    steps = {}
    for step in raw.get('steps', []):
        path = os.path.join(output_dir, step.get('id', ''), STEP_RESULT)
        steps[step.get('id')] = _read_json(path) if os.path.isfile(path) else None
    return {'output_dir': output_dir, 'workflow': raw,
            'result': _read_json(result_path) if os.path.isfile(result_path) else None,
            'steps': steps}
