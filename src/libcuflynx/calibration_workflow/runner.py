'''
Running a calibration workflow: every step through the normal path of its kind (param_id,
sensitivity analysis, emulator training, identifiability analysis), in dependency order, then
merging the calibrations' results into the workflow's target.

Output layout (``output_dir``)::

    calibration_workflow.json      the workflow as run
    workflow_result.json           order, per-step provenance, the merged parameter set
    <step_id>/
        step_result.json           calibrated values, fixed values and priors used, input hashes
        model/                     the generated step model (resources/, generated_models/)
        param_id/                  the normal param_id outputs (best_param_vals.npy, ...)
        <step_id>_params_for_id.json  a reduction's choice, or a step's own with the
                                   parameters priors_from adds
        distribution/              only with store_distribution (see distributions.py)
        sensitivity/               a sensitivity step's outputs (sa_options.output_dir)
        emulator/                  an emulation step's emulator
        <step_id>_laplace_*.npy    an identifiability step's Laplace mean and covariance
    _target/model/                 the target generated with every calibrated value

A step's model is generated from a library module or from a copy of its module array; the
files the workflow names are only read.

Under MPI every rank runs every step (calibration is parallel); generation and every file
write happen on rank 0, and what rank 0 reads is broadcast.
'''

import copy
import csv
import datetime
import glob
import hashlib
import json
import os
import shutil
import time
import warnings

import numpy as np

from libcuflynx.calibration_workflow import naming
from libcuflynx.utilities import cuflynx_study
from libcuflynx.calibration_workflow.distributions import (DISTRIBUTION_DIR,
                                                           StoredDistribution,
                                                           parameters_from_info)
from libcuflynx.calibration_workflow.generate import (UnsetModelParameters, generate_model,
                                                      read_parameter_rows)
from libcuflynx.calibration_workflow.resolve import (CalibratedParameter,
                                                     read_calibrated_parameters,
                                                     resolve_workflow)
from libcuflynx.calibration_workflow.select import (UnsetParametersError, protocol_values,
                                                    step_input_paths, write_step_inputs)
from libcuflynx.sensitivity_analysis.parameter_reduction import SobolReduction
from libcuflynx.calibration_workflow.spec import (WORKFLOW_FILE_NAME, WorkflowError, load_workflow,
                                                  uses_log_posterior)

STEP_RESULT = 'step_result.json'
# where a step's calibrated values come from
POINT_BEST_FIT = 'best_fit'
POINT_POSTERIOR_MEDIAN = 'posterior_median'
WORKFLOW_RESULT = 'workflow_result.json'
TARGET_DIR = '_target'
SENSITIVITY_DIR = 'sensitivity'
EMULATOR_DIR = 'emulator'
SOBOL_INDICES_GLOB = 'all_outputs_n*_Sobol_indices.csv'
#: the result keys the study manifest lists as each kind's outputs
STEP_OUTPUTS = {'calibration': ('param_id_output_dir', 'distribution_dir'),
                'sensitivity': ('sensitivity_dir', 'sobol_indices'),
                'reduction': ('params_for_id',),
                'emulation': ('emulator_dir',),
                'identifiability': ('laplace',)}

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
        rs = self.resolved.steps[step_id]
        if rs.model is None:
            return {}       # a reduction: its inputs are its sensitivity step's
        files = {'obs_data': rs.obs_data_path,
                 'params_for_id': self.params_for_id_of(rs.step)}
        if not os.path.isfile(files['params_for_id']):
            del files['params_for_id']    # a reduction's, not yet written
        if os.path.isfile(rs.model.parameters_path):
            files['parameters'] = rs.model.parameters_path
        if rs.model.kind == 'array':
            files['module_array'] = rs.model.model.module_array
        if rs.selection is not None:
            # a selection starts from the target's values
            files['target_parameters'] = self.workflow.target.parameters
            files['target_module_array'] = self.workflow.target.module_array
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

    def fixed_values(self, step, sources):
        '''``{model name in this step: row}`` from every step in ``sources``: its calibrated
        values and the values it was itself given as fixed.'''
        values = {}
        for source in sources:
            result = self.require_current(source, step.id)
            vessel_of = self.resolved.map_between(source, step.id)
            left_out = []
            for record in result['calibrated'] + result['fixed']:
                vessel = vessel_of(record['vessel'])
                if vessel is None:
                    left_out.append(record['model_name'])
                    continue
                name = naming.model_name(vessel, record['param'])
                origin = record.get('from_step', source)
                if name in values and not np.isclose(values[name]['value'], record['value'],
                                                      rtol=1e-12, atol=0.0):
                    raise WorkflowError(
                        f'step "{step.id}" is given two different fixed values for "{name}": '
                        f'{values[name]["value"]} (from "{values[name]["from_step"]}") and '
                        f'{record["value"]} (from "{origin}").')
                values[name] = {'vessel': vessel,
                                'param': record['param'], 'model_name': name,
                                'host_name': naming.host_row_name(vessel, record['param']),
                                'value': record['value'], 'units': record.get('units', ''),
                                'from_step': origin}
            if left_out:
                warnings.warn(f'step "{step.id}": {len(left_out)} value(s) of "{source}" are '
                              f'not fixed, because {step.model.label()} has no such vessel: '
                              f'{sorted(left_out)}.', UserWarning, stacklevel=2)
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
            try:
                distribution = distribution.mapped_with(
                    self.resolved.map_between(prior.step, step.id), step.model.label())
            except ValueError as exc:
                raise WorkflowError(f'step "{step.id}" takes priors from "{prior.step}": '
                                    f'{exc}') from exc
            out.append((prior, distribution))
        return out

    # -- one step --------------------------------------------------------------------------

    def run_step(self, step, index, total):
        started, t0 = _now(), time.time()
        self.emit('step_started', step=step.id, index=index, total=total, kind=step.kind,
                  model=step.model.as_dict() if step.model is not None else None)
        run = {'calibration': self._run_calibration, 'sensitivity': self._run_sensitivity,
               'reduction': self._run_reduction, 'emulation': self._run_emulation,
               'identifiability': self._run_identifiability}[step.kind]
        record = run(step, self.workflow.step_settings(step))

        def finish():
            result = dict({'step': step.id, 'kind': step.kind, 'status': 'done',
                           'model': step.model.as_dict() if step.model is not None else None,
                           'place': self.resolved.steps[step.id].place if step.model else '',
                           'started': started, 'finished': _now(),
                           'duration_s': round(time.time() - t0, 3),
                           'calibrated': [], 'fixed': [], 'inputs': self.inputs(step.id)},
                          **record)
            _write_json(os.path.join(self.step_dir(step.id), STEP_RESULT), result)
            return result

        result = self.on_root(finish)
        self.emit('step_finished', step=step.id, index=index, total=total, kind=step.kind,
                  best_cost=result.get('best_cost'), duration_s=result['duration_s'])
        return result

    def params_for_id_of(self, step):
        '''The params_for_id a step runs with: its own, or the one its reduction wrote.'''
        if step.params_for_id_from:
            return reduced_params_for_id_path(self.output_dir, step.params_for_id_from)
        return step.params_for_id

    def _prepare(self, step, settings):
        '''Rank 0: a clean step directory and the step's model, generated with the values it is
        given (``fixed_from``, and ``result_from`` including the parameters it analyses), or
        copied from its ``emulator_from`` step. Returns ``(generated, params_for_id, units,
        fixed, priors)``.'''
        resolved = self.resolved.steps[step.id]
        step_dir = self.step_dir(step.id)
        params_for_id = self.params_for_id_of(step)
        if step.params_for_id_from:
            self.require_current(step.params_for_id_from, step.id)
        own = read_calibrated_parameters(params_for_id, step.id)
        if resolved.renames_inputs:
            own = [CalibratedParameter(resolved.vessel_map.from_target(p.vessel) or p.vessel,
                                       p.param) for p in own]
        own = {p.model_name for p in own}
        # a parameter the step calibrates itself is not fixed: the later calibration wins
        fixed = {n: row for n, row in self.fixed_values(step, step.fixed_from).items()
                 if n not in own}
        if step.result_from:
            fixed.update(self.fixed_values(step, [step.result_from]))
        priors = self.priors(step)
        prior_names = [n for _, dist in priors for n in dist.names]
        clash = sorted((own | set(fixed)) & set(prior_names))
        if clash:
            raise WorkflowError(f'step "{step.id}" is given priors for {clash}, which it '
                                f'also calibrates or is given as fixed values.')
        if os.path.isdir(step_dir):
            shutil.rmtree(step_dir)
        os.makedirs(step_dir)
        obs_data = resolved.obs_data_path
        if resolved.renames_inputs:
            obs_data, params_for_id = write_step_inputs(
                step_dir, step, params_for_id, resolved.vessel_map, self.resolved.target.vessels)
        if step.emulator_from:
            source = self.require_current(step.emulator_from, step.id)
            source_dir = self.step_dir(step.emulator_from)
            shutil.copytree(os.path.join(source_dir, 'model'), os.path.join(step_dir, 'model'))
            generated = {k: (v.replace(source_dir, step_dir, 1) if isinstance(v, str) else v)
                         for k, v in source['generated'].items()}
        else:
            overrides = [{'variable_name': row['host_name'], 'units': row['units'],
                          'value': repr(float(row['value'])),
                          'data_reference': f'fixed by calibration workflow '
                                            f'{self.workflow.name}, step {row["from_step"]}'}
                         for name, row in sorted(fixed.items())]
            generated = generate_step_model(
                resolved, os.path.join(step_dir, 'model'), step.id,
                under(resolved.base_rows, overrides), obs_data, self.resolved.library_inputs,
                settings)
        model_rows = {r['variable_name']: r
                      for r in read_parameter_rows(generated['parameters_path'])}
        for name, row in fixed.items():
            if name not in model_rows:
                raise WorkflowError(
                    f'step "{step.id}": the value fixed from "{row["from_step"]}" for '
                    f'"{name}" has no parameter of that name in the step model '
                    f'({generated["parameters_path"]}).')
        if priors:
            source_path = params_for_id
            params_for_id = os.path.join(step_dir, f'{step.id}_params_for_id.json')
            _write_params_for_id_with_priors(source_path, params_for_id, priors, step.id)
        units = {k: v.get('units', '') for k, v in model_rows.items()}
        return generated, params_for_id, units, fixed, priors

    def _inp(self, step, settings, generated, params_for_id):
        '''The user_inputs of one step's run. A step on an emulator runs the emulation step's
        model (its files carry that step's prefix) with its emulator.'''
        step_dir = self.step_dir(step.id)
        prefix = step.emulator_from or step.id
        resolved = self.resolved.steps[step.id]
        obs_data = (step_input_paths(step_dir, step)[0] if resolved.renames_inputs
                    else resolved.obs_data_path)
        inp = copy.deepcopy(STEP_DEFAULTS)
        inp.update(copy.deepcopy(settings))
        inp.update({
            'file_prefix': prefix,
            'study_name': step.id,
            'resources_dir': generated['resources_dir'],
            'generated_models_dir': generated['generated_models_dir'],
            'input_param_file': f'{prefix}_parameters.json',
            'param_id_obs_path': obs_data,
            'params_for_id_file': params_for_id,
            'param_id_output_dir': os.path.join(step_dir, 'param_id'),
            'do_uq': step.store_distribution,
        })
        inp.update({k: v for k, v in self.resolved.library_inputs.items()})
        if step.emulator_from:
            emulation = self.workflow.step(step.emulator_from)
            inp['use_emulator'] = True
            inp['emulator_settings'] = dict(
                self.workflow.step_settings(emulation).get('emulator_settings') or {},
                emulator_dir=emulator_dir(self.output_dir, emulation.id))
        return inp

    def _run_stage(self, step, what, fn, *args):
        try:
            return fn(*args)
        except SystemExit as exc:
            # the user_inputs parser exit()s on a bad configuration, after printing why
            raise WorkflowStepError(step.id, f'the {what} configuration was refused (exit '
                                             f'{exc.code}); the messages above say why.')

    def _run_calibration(self, step, settings):
        from libcuflynx.scripts.param_id_run_script import run_param_id

        resolved = self.resolved.steps[step.id]
        generated, params_for_id, units, fixed, priors = self.on_root(
            lambda: self._prepare(step, settings))
        inp = self._inp(step, settings, generated, params_for_id)
        if priors:
            inp['joint_priors'] = [([n for n in dist.names],
                                    dist.as_prior(prior.kind, prior.inflate))
                                   for prior, dist in priors]
        outcome = self._run_stage(step, 'param_id', run_param_id, inp)

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
                    parameter = CalibratedParameter(vessel=vessel, param=param)
                    calibrated.append({
                        'vessel': vessel, 'param': param, 'model_name': parameter.model_name,
                        'target_model_name': resolved.target_model_name(parameter),
                        'target_host_name': naming.host_row_name(resolved.to_target(vessel),
                                                                 param),
                        'target_parameterisation_name':
                            resolved.target_parameterisation_name(parameter),
                        'value': float(value), 'units': units.get(parameter.model_name, '')})
            return {
                'method': settings.get('param_id_method'), 'best_cost': best_cost,
                'point_estimate': point_estimate,
                'best_fit': [float(v) for v in best],
                'calibrated': calibrated,
                'fixed': [dict(row) for _, row in sorted(fixed.items())],
                'priors': [dict(prior.as_dict(), parameters=dist.names)
                           for prior, dist in priors],
                'params_for_id_used': params_for_id,
                'settings': settings,
                'generated': generated,
                'model_path': generated['model_path'],
                'param_id_output_dir': case_dir,
                'distribution_dir': distribution_dir,
            }

        return self.on_root(finish)

    def _run_sensitivity(self, step, settings):
        from libcuflynx.scripts.sensitivity_analysis_run_script import run_SA

        generated, params_for_id, _, fixed, _ = self.on_root(
            lambda: self._prepare(step, settings))
        out_dir = os.path.join(self.step_dir(step.id), SENSITIVITY_DIR)
        sa_options = dict(settings['sa_options'], output_dir=out_dir)
        inp = self._inp(step, settings, generated, params_for_id)
        inp['sa_options'] = sa_options
        self._run_stage(step, 'sensitivity analysis', run_SA, inp)

        def finish():
            method = sa_options.get('method', 'sobol')
            found = sorted(glob.glob(os.path.join(out_dir, SOBOL_INDICES_GLOB)))
            if method == 'sobol' and not found:
                raise WorkflowStepError(step.id, f'the Sobol analysis wrote no indices into '
                                                 f'{out_dir}.')
            return {'method': method, 'sensitivity_dir': out_dir,
                    'sobol_indices': found[0] if found else None,
                    'fixed': [dict(row) for _, row in sorted(fixed.items())],
                    'params_for_id_used': params_for_id, 'settings': settings,
                    'generated': generated, 'model_path': generated['model_path']}

        return self.on_root(finish)

    def _run_reduction(self, step, settings):
        def reduce():
            source = self.require_current(step.sensitivity_from, step.id)
            reduction = SobolReduction.from_options(step.reduction)
            step_dir = self.step_dir(step.id)
            if os.path.isdir(step_dir):
                shutil.rmtree(step_dir)
            os.makedirs(step_dir)
            out = reduced_params_for_id_path(self.output_dir, step.id)
            try:
                ranked = reduction.reduce(source['sobol_indices'],
                                          source['params_for_id_used'], out)
            except ValueError as exc:
                raise WorkflowStepError(step.id, f'reducing "{step.sensitivity_from}": '
                                                 f'{exc}') from exc
            return {'sensitivity_step': step.sensitivity_from,
                    'reduction': reduction.as_dict(),
                    'selected': [{'parameter': name, 'score': float(score)}
                                 for name, score in ranked],
                    'kept_rows': len(ranked), 'params_for_id': out}

        return self.on_root(reduce)

    def _run_emulation(self, step, settings):
        from libcuflynx.scripts.train_emulator_run_script import train_emulator

        generated, params_for_id, _, fixed, _ = self.on_root(
            lambda: self._prepare(step, settings))
        inp = self._inp(step, settings, generated, params_for_id)
        directory = emulator_dir(self.output_dir, step.id)
        inp['emulator_settings'] = dict(settings.get('emulator_settings') or {},
                                        emulator_dir=directory)
        self._run_stage(step, 'emulator training', train_emulator, inp)

        def finish():
            return {'emulator_dir': directory,
                    'fixed': [dict(row) for _, row in sorted(fixed.items())],
                    'params_for_id_used': params_for_id, 'settings': settings,
                    'generated': generated, 'model_path': generated['model_path']}

        return self.on_root(finish)

    def _run_identifiability(self, step, settings):
        from libcuflynx.scripts.identifiability_run_script import run_identifiability_analysis
        from libcuflynx.parsers.PrimitiveParsers import ObsAndParamDataParser

        generated, params_for_id, units, fixed, _ = self.on_root(
            lambda: self._prepare(step, settings))
        info = ObsAndParamDataParser().get_param_id_info(params_for_id)
        model_rows = self.on_root(lambda: {r['variable_name']: float(r['value'])
                                           for r in read_parameter_rows(
                                               generated['parameters_path'])})
        # the analysis is at the result_from step's values, which the model was generated with
        best = []
        for qnames in info['param_names']:
            first = qnames[0] if isinstance(qnames, (list, tuple)) else qnames
            vessel, _, param = str(first).partition('/')
            best.append(model_rows[CalibratedParameter(vessel=vessel, param=param).model_name])
        inp = self._inp(step, settings, generated, params_for_id)
        inp['ia_options'] = settings['ia_options']
        self._run_stage(step, 'identifiability analysis', run_identifiability_analysis, inp,
                        best)

        def finish():
            record = {'method': settings['ia_options'].get('method'),
                      'parameters': list(info.get('param_labels') or []),
                      'at': [float(v) for v in best],
                      'fixed': [dict(row) for _, row in sorted(fixed.items())],
                      'params_for_id_used': params_for_id, 'settings': settings,
                      'generated': generated, 'model_path': generated['model_path']}
            prefix = os.path.join(self.step_dir(step.id), step.emulator_from or step.id)
            mean, covariance = prefix + '_laplace_mean.npy', prefix + '_laplace_covariance.npy'
            if os.path.isfile(covariance):
                cov = np.atleast_2d(np.load(covariance))
                record['laplace'] = {'mean': np.atleast_1d(np.load(mean)).tolist(),
                                     'covariance': cov.tolist(),
                                     'std': np.sqrt(np.diag(cov)).tolist(),
                                     'mean_path': mean, 'covariance_path': covariance}
            return record

        return self.on_root(finish)

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
                    'model': step.model.as_dict(),
                    'obs_data_sha256': sha256(step.obs_data),
                    'chain': chain_path, 'created': _now()})
        return distribution.save(os.path.join(self.step_dir(step.id), DISTRIBUTION_DIR),
                                 chain=chain)

    # -- the merged result -----------------------------------------------------------------

    # -- the study manifest ------------------------------------------------------------------

    def _writes_study(self):
        return self.workflow.settings.get('write_cuflynx_study', True) is not False

    def _study_inp(self, directory, name):
        return {'param_id_output_dir': directory, 'study_name': name,
                'write_cuflynx_study': self._writes_study()}

    def study_manifest(self):
        '''The workflow's own manifest: ``<output_dir>/<workflow_name>_cuflynx_study.json``.'''
        return cuflynx_study.manifest_path(self._study_inp(self.output_dir, self.workflow.name))

    def step_manifest(self, step_id):
        '''The manifest a step's calibration writes, in its param_id output directory.'''
        return cuflynx_study.manifest_path(
            self._study_inp(os.path.join(self.step_dir(step_id), 'param_id'), step_id))

    def record_study(self, order, failed=None, workflow_result=None):
        '''Write the workflow's manifest, with every step's state so far, and link each step
        that has run back to it. Rank 0; a failure is a warning, as for any manifest.'''
        if self.rank != 0 or not self._writes_study():
            return
        result_path = os.path.join(self.output_dir, WORKFLOW_RESULT)
        steps = []
        for step in order:
            rs = self.resolved.steps[step.id]
            result = self.step_result(step.id)
            done = bool(result) and result.get('status') == 'done'
            entry = {
                'id': step.id, 'kind': step.kind, 'model': _model_files(step.model),
                'obs_data': rs.obs_data_path, 'params_for_id': step.params_for_id,
                'place': rs.place if step.model is not None else '',
                'status': 'failed' if step.id == failed else ('done' if done else 'not_run'),
                'run': (result or {}).get('param_id_output_dir') if done else None,
                'outputs': {k: (result or {}).get(k) for k in STEP_OUTPUTS[step.kind]}
                if done else None,
                'manifest': self.step_manifest(step.id) if done else None}
            entry.update({key: ids for key, ids in step.sources().items() if ids or
                          key in ('fixed_from', 'priors_from')})
            if step.kind == 'reduction':
                entry['reduction'] = dict(step.reduction)
            steps.append(entry)
        target = (workflow_result or {}).get('target_model') or {}
        block = {'name': self.workflow.name,
                 'file': os.path.join(self.output_dir, WORKFLOW_FILE_NAME),
                 'source': self.workflow.path,
                 'result': result_path if os.path.isfile(result_path) else None,
                 'complete': all(s['status'] == 'done' for s in steps),
                 'target': {'model': _model_files(self.workflow.target),
                            'generated_model': target.get('model_path'),
                            'calibrated_parameters':
                                (workflow_result or {}).get('calibrated_parameters_file')},
                 'steps': steps}
        inp = self._study_inp(self.output_dir, self.workflow.name)
        if target.get('model_path'):
            inp['model_path'] = target['model_path']
        manifest = cuflynx_study.update_study(inp, workflow=block)
        for entry in steps:
            if manifest and entry['manifest'] and os.path.isfile(entry['manifest']):
                cuflynx_study.update_study(
                    self._study_inp(os.path.dirname(entry['manifest']), entry['id']),
                    workflow={'manifest': manifest, 'step': entry['id']})

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
                    'host_name': record['target_host_name'],
                    'parameterisation_name': record['target_parameterisation_name'],
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
                'status': r.get('status'), 'model': r.get('model'),
                'place': r.get('place'), 'started': r.get('started'),
                'finished': r.get('finished'), 'best_cost': r.get('best_cost'),
                'inputs': r.get('inputs'), 'result': os.path.join(sid, STEP_RESULT)})
                for sid, r in results.items()},
            'merged_parameters': sorted(merged.values(), key=lambda r: r['model_name']),
            'target_model': None,
            'calibrated_parameters_file': None,
        }
        if generate_target and merged:
            overrides = [{'variable_name': r['host_name'], 'units': r['units'],
                          'value': repr(float(r['value'])),
                          'data_reference': f'calibration workflow {self.workflow.name}, '
                                            f'step {r["from_step"]}'}
                         for r in workflow_result['merged_parameters']]
            target_dir = os.path.join(self.output_dir, TARGET_DIR)
            if os.path.isdir(target_dir):
                shutil.rmtree(target_dir)
            generated = generate_model(
                self.workflow.target, os.path.join(target_dir, 'model'), 'target', overrides,
                library_inputs=self.resolved.library_inputs,
                model_type=self.workflow.settings.get('model_type', 'cellml'),
                solver=self.workflow.settings.get('solver'))
            names = {r['variable_name'] for r in read_parameter_rows(generated['parameters_path'])}
            missing = sorted(r['model_name'] for r in workflow_result['merged_parameters']
                             if r['model_name'] not in names)
            if missing:
                raise WorkflowError(
                    f'the target {self.workflow.target.label()} has no parameter {missing}, '
                    f'which the steps calibrated. Name where each step\'s model sits in the '
                    f'target: "submodule" or "vessel" for a library module, "vessel_map" for a '
                    f'module array.')
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


def under(base_rows, overrides):
    '''``base_rows`` without those ``overrides`` set again, then ``overrides``: the rows a step
    model is generated with.'''
    replaced = {naming.host_model_name(r['variable_name']) for r in overrides}
    return [r for r in base_rows
            if naming.host_model_name(r['variable_name']) not in replaced] + list(overrides)


def generate_step_model(resolved_step, work_dir, prefix, rows, obs_data, library_inputs,
                        settings):
    '''Generate the model of ``resolved_step`` with parameter ``rows``; the step's protocol
    (``obs_data``) gives the parameters nothing else does. Parameters still without a value are
    an UnsetParametersError naming them as the target's parameters file does.'''
    try:
        return generate_model(
            resolved_step.model.model, work_dir, prefix, rows, library_inputs=library_inputs,
            model_type=settings.get('model_type', STEP_DEFAULTS['model_type']),
            solver=settings.get('solver'), protocol=protocol_values(obs_data),
            instance_values=(resolved_step.selection.instance_values
                             if resolved_step.selection else None))
    except UnsetModelParameters as exc:
        vessels = sorted(resolved_step.model.vessels, key=len, reverse=True)
        names = []
        for name in exc.model_names:
            vessel = next((v for v in vessels if name.endswith('_' + v)), None)
            if vessel is None:
                names.append(name)      # a global
                continue
            names.append(naming.host_row_name(resolved_step.to_target(vessel),
                                              name[:-len(vessel) - 1]))
        raise UnsetParametersError(resolved_step.step.id, names) from exc


def _model_files(model):
    '''A step's or target's model as the manifest records it: a module array's files, or a
    library module's ``{module_type, version, parameterisation}``; None for a reduction.'''
    return model.as_dict() if model is not None else None


def reduced_params_for_id_path(output_dir, step_id):
    '''The params_for_id a reduction step writes.'''
    return os.path.join(output_dir, step_id, f'{step_id}_params_for_id.json')


def emulator_dir(output_dir, step_id):
    '''Where an emulation step's emulator is trained.'''
    return os.path.join(output_dir, step_id, EMULATOR_DIR)


def write_calibrated_parameters_to(source, output_dir, path, module_library_dirs=None):
    '''Write the target's parameters with the merged values of the workflow run in
    ``output_dir`` to ``path``, as :func:`write_calibrated_parameters` does; returns ``path``.
    Only ``path`` is written: the target's own files are not changed.

    Reads the run's ``workflow_result.json``, so a run made one step at a time (``only``,
    ``from_step``) can be written once every step has a result. WorkflowError when there is no
    result, or a step has not run or is stale (its inputs changed since it ran).'''
    workflow = source if hasattr(source, 'steps') else load_workflow(source, module_library_dirs)
    result_path = os.path.join(output_dir, WORKFLOW_RESULT)
    if not os.path.isfile(result_path):
        raise WorkflowError(f'{output_dir} holds no run of workflow "{workflow.name}" '
                            f'({WORKFLOW_RESULT} is missing).')
    status = workflow_status(workflow, output_dir)
    waiting = [s for s, v in status['steps'].items() if v['status'] != 'done' or v['stale']]
    if waiting:
        raise WorkflowError(f'workflow "{workflow.name}": steps {waiting} have no current result '
                            f'in {output_dir} (not run, or their inputs changed since); run '
                            f'them before writing the calibrated parameters.')
    merged = _read_json(result_path)['merged_parameters']
    return write_calibrated_parameters(resolve_workflow(workflow).target, merged, workflow.name,
                                       path=path)


def write_calibrated_parameters(target, merged_parameters, workflow_name, path=None):
    '''``path`` (default ``<name>_calibrated_parameters.json`` beside the target's parameters
    file, ``<name>`` its parameterisation, or its module array's prefix): the target's parameters
    file (nested JSON, utilities/parameters_file.py) with each merged value set at the path the
    target names it by -- ``<vessel>/<variable>`` in a module array's file (a vessel inside a
    super instance, ``sys__C/q``, at ``sys/C/q``), the module's own path in a parameterisation --
    and added where the file lacks it. Only ``path`` is written; the target's files are not
    changed.'''
    from libcuflynx.utilities import parameters_file as pf
    kind = pf.MODEL if target.kind == 'array' else pf.MODULE
    doc = pf.read_parameters_file(target.parameters_path, kind=kind)
    today = datetime.date.today().isoformat()
    for record in merged_parameters:
        name = record['host_name'] if target.kind == 'array' else record['parameterisation_name']
        place = pf.name_path(name, kind)
        leaf = dict(pf.get_leaf(doc, place) or {'units': record.get('units', '')})
        leaf.update(value='%.10g' % record['value'],
                    data_reference=f'calibrated by workflow {workflow_name}, step '
                                   f'{record["from_step"]} ({today})')
        if 'sourced' in leaf:
            leaf['sourced'] = 'no'
        pf.set_leaf(doc, place, leaf)
    return pf.write_parameters_file(path or target.file('calibrated_parameters.json'), doc)


def _write_params_for_id_with_priors(source_path, out_path, priors, step_id):
    '''The step's own params_for_id plus one ``uniform`` entry (for the bounds) per parameter a
    prior is given for; the prior's density itself is added as a joint prior.'''
    from libcuflynx.utilities import params_for_id_file as pfi
    doc = pfi.read_params_for_id(source_path, f'step "{step_id}": ', keep_text=True)
    entries = list(doc.get('params') or [])
    for prior, distribution in priors:
        for par in distribution.parameters:
            if not (np.isfinite(par.min) and np.isfinite(par.max)):
                raise WorkflowError(f'step "{step_id}": the prior from "{prior.step}" over '
                                    f'{par.model_names()} has no finite bounds.')
            entries.append({'targets': [pfi.target_name(v, p) for v, p in par.targets],
                            'min': par.min, 'max': par.max,
                            'name_for_plotting': par.model_names()[0], 'prior': 'uniform'})
    pfi.write_params_for_id(out_path, dict(doc, params=entries))


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
            vessel_of = resolved.map_between(source_id, step.id)
            for p in resolved.steps[source_id].parameters:
                vessel = vessel_of(p.vessel)
                fixed.append({'from_step': source_id, 'model_name': p.model_name,
                              'in_step': naming.model_name(vessel, p.param) if vessel else None})
        entry = {
            'id': step.id, 'kind': step.kind,
            'model': step.model.as_dict() if step.model is not None else None,
            'place': rs.place if step.model is not None else '',
            'depends_on': list(step.depends_on),
            'obs_data': rs.obs_data_path,
            'params_for_id': step.params_for_id,
            'calibrates': [{'model_name': p.model_name,
                            'target_model_name': rs.target_model_name(p),
                            'target_parameterisation_name': rs.target_parameterisation_name(p)}
                           for p in rs.parameters] if step.kind == 'calibration' else [],
            'fixed_from': fixed,
            'priors_from': [p.as_dict() for p in step.priors_from],
            'store_distribution': step.store_distribution,
            'settings': workflow.step_settings(step) if step.model is not None else {},
        }
        for key in ('emulator_from', 'result_from', 'sensitivity_from', 'params_for_id_from'):
            if getattr(step, key):
                entry[key] = getattr(step, key)
        if rs.selection is not None:
            # what the selection runs as: its library module at one instance, or the cut array
            entry['selection'] = (
                {'runs_as': 'library_module', 'module': rs.selection.model.as_dict(),
                 'vessel': rs.selection.vessel} if rs.selection.vessel else
                {'runs_as': 'module_array',
                 'records': [r['name'] for r in rs.selection.model.records]})
        if step.kind == 'reduction':
            entry['reduction'] = dict(step.reduction)
        steps.append(entry)
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
        write_calibrated: write the merged set as the target parameterisation's
            ``<parameterisation>_calibrated_parameters.json`` (complete runs only).
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
    run.record_study(order)
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
            run.record_study(order, failed=step.id)
            raise
        run.record_study(order)
    result = run.on_root(lambda: run.merge(order, write_calibrated=write_calibrated))
    run.record_study(order, workflow_result=result)
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
            for source_id in step.depends_on:
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
    inputs changed since they ran. obs_data / params_for_id are None when the parameterisation has
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

    # overrides: {host parameters file name: (record, step id)}
    overrides, fixed, calibrated, base_rows = {}, [], [], []
    if view == TARGET_VIEW:
        kind, model = 'target', workflow.target
        latest = {}
        for step in workflow.topological_order():
            result = available(step.id)
            for record in (result or {}).get('calibrated', []):
                latest[record['target_model_name']] = (record, step.id)
        overrides = {rec['target_host_name']: (rec, sid) for rec, sid in latest.values()}
        calibrated = [{'model_name': name, 'value': rec['value'], 'units': rec.get('units', ''),
                       'from_step': sid} for name, (rec, sid) in sorted(latest.items())]
        own_step = next((s for s in workflow.steps if s.model == workflow.target
                         and s.kind == 'calibration'), None)
        place = ''
        own_result = run.step_result(own_step.id) if own_step else None
    else:
        step = workflow.step(view)
        if step.model is None:
            raise WorkflowError(f'step "{step.id}" is a reduction, which has no model.')
        kind, model = 'step', step.model
        rs = resolved.steps[step.id]
        place = rs.place
        base_rows = rs.base_rows
        for source_id in step.fixed_from:
            result = available(source_id)
            if result is None:
                continue
            vessel_of = resolved.map_between(source_id, step.id)
            for record in result['calibrated'] + result['fixed']:
                vessel = vessel_of(record['vessel'])
                if vessel is None:
                    continue
                name = naming.model_name(vessel, record['param'])
                origin = record.get('from_step', source_id)
                overrides[naming.host_row_name(vessel, record['param'])] = (
                    {'value': record['value'], 'units': record.get('units', '')}, origin)
                fixed.append({'model_name': name, 'value': record['value'],
                              'units': record.get('units', ''), 'from_step': origin})
        own_step = step
        own_result = run.step_result(step.id)
        if own_result and own_result.get('status') == 'done':
            calibrated = [{'model_name': r['model_name'], 'value': r['value'],
                           'units': r.get('units', ''), 'from_step': step.id}
                          for r in own_result['calibrated']]
            if with_own_results:
                for r, record in zip(own_result['calibrated'], calibrated):
                    overrides.setdefault(naming.host_row_name(r['vessel'], r['param']),
                                         (record, step.id))
    rows = [{'variable_name': name, 'units': rec.get('units', ''),
             'value': repr(float(rec['value'])),
             'data_reference': f'calibration workflow {workflow.name}, step {sid}'}
            for name, (rec, sid) in sorted(overrides.items())]
    view_dir = os.path.join(os.path.abspath(work_dir), view)
    if os.path.isdir(view_dir):
        shutil.rmtree(view_dir)
    settings = workflow.step_settings(own_step) if kind == 'step' else workflow.settings
    has_own = own_step is not None
    obs_data = own_step.obs_data if has_own else None
    params_for_id = run.params_for_id_of(own_step) if has_own else None
    if kind == 'step':
        if rs.renames_inputs and os.path.isfile(params_for_id):
            os.makedirs(view_dir)
            obs_data, params_for_id = write_step_inputs(view_dir, own_step, params_for_id,
                                                        rs.vessel_map, resolved.target.vessels)
        generated = generate_step_model(rs, view_dir, view, under(base_rows, rows), obs_data,
                                        resolved.library_inputs, settings)
    else:
        generated = generate_model(
            model, view_dir, view, rows, library_inputs=resolved.library_inputs,
            model_type=settings.get('model_type', STEP_DEFAULTS['model_type']),
            solver=settings.get('solver'))

    def existing(path):
        return path if path and os.path.isfile(path) else None

    return {
        'view': view, 'kind': kind, 'model': model.as_dict(),
        'step_id': own_step.id if has_own else None,
        'place': place,
        'model_path': generated['model_path'],
        'flat_model_path': generated['flat_model_path'],
        'parameters_path': generated['parameters_path'],
        'obs_data_path': existing(obs_data),
        'params_for_id_path': existing(params_for_id),
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
