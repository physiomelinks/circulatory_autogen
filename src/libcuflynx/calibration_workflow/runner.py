'''
Running a calibration workflow: every step through the normal path of its kind (param_id,
sensitivity analysis, emulator training, identifiability analysis, a parameter sweep or one
simulation), in
dependency order, then merging the calibrations' results and the values set by hand into the
workflow's target.

Output layout (``output_dir``)::

    calibration_workflow.json      the workflow as run
    workflow_result.json           order, per-step provenance, the merged parameter set, the
                                   commits and libCellML version it ran with
    provenance_graph.json          how every value was set (reporting/provenance.py)
    provenance.svg                 that graph as a diagram
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
        param_id/profile_likelihood/  an identifiability step's profiles (method
                                   profile_likelihood)
        <step_id>_calibrated.cellml  a calibration's model with its best fit (a CellML model)
        uq/                        a uq step's run (its chain, or its Laplace mean and
                                   covariance); its distribution is in distribution/
        sweep/                     a sweep or simulate step's runs (sweep.py)
        figure/                    a figure step's files (figures.py)
    _target/model/                 the target generated with every merged value
    models/<view>/                 models generated for viewing (workflow_model with this
                                   work_dir)
    inputs/                        the study's copies of its inputs (cuflynx_study)

No step may be named ``models``, ``target`` (the target's view), ``_target`` or ``inputs``.

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
import tempfile
import time
import warnings

import numpy as np

from libcuflynx.errors import CAInputError
from libcuflynx.calibration_workflow import figures, naming, sweep as sweep_runs
from libcuflynx.parsers.PrimitiveParsers import DEFAULT_DO_AD, DEFAULT_MODEL_TYPE
from libcuflynx.utilities import cuflynx_study
from libcuflynx.calibration_workflow.distributions import (DISTRIBUTION_DIR,
                                                           StoredDistribution,
                                                           parameters_from_info)
from libcuflynx.calibration_workflow.generate import (UnsetModelParameters, generate_model,
                                                      read_parameter_rows)
from libcuflynx.calibration_workflow.protocol_factors import has_factors, resolve_factors
from libcuflynx.calibration_workflow.resolve import (CalibratedParameter,
                                                     read_calibrated_parameters,
                                                     resolve_workflow,
                                                     check_merged_once)
from libcuflynx.calibration_workflow.select import (UnsetParametersError, protocol_values,
                                                    step_input_paths, write_step_inputs)
from libcuflynx.sensitivity_analysis.parameter_reduction import ParamReduction
from libcuflynx.sensitivity_analysis.parameter_reduction import REPORT_FILE as REDUCTION_REPORT_FILE
from libcuflynx.calibration_workflow.spec import (INHERITED_BASE, SWEEP_KINDS,
                                                  WORKFLOW_FILE_NAME, WorkflowError,
                                                  load_workflow, refuse_older_step_keys,
                                                  uq_method, uq_options as _uq_options,
                                                  uses_log_posterior)
from libcuflynx.utilities.parameters_file import ORIGINS
from libcuflynx.utilities.mpi_utils import run_on_root
from libcuflynx.utilities.paths import git_state, repo_root

STEP_RESULT = 'step_result.json'
# the inputs of a step that are not one file: their digests are in its recorded inputs
SETTINGS_INPUT = 'settings'
#: the inputs that are a step's user function files, and the settings naming them
FUNCS_INPUTS = {'operation_funcs': 'operation_funcs_external_path',
                'cost_funcs': 'cost_funcs_external_path',
                'modifier_funcs': 'modifier_funcs_external_path'}
LIBRARY_INPUT = 'module_library'
#: the files of a module library a model is generated from
LIBRARY_FILE_SUFFIXES = ('.cellml', '_modules_config.json', '_parameters.json')
# where a step's calibrated values come from
POINT_BEST_FIT = 'best_fit'
POINT_POSTERIOR_MEDIAN = 'posterior_median'
WORKFLOW_RESULT = 'workflow_result.json'
TARGET_DIR = '_target'
#: the folder of a run's output directory that workflow_model may generate into; the schema
#: refuses it as a step id
MODELS_DIR = 'models'
# the file prefix the target's model is generated with
TARGET_PREFIX = 'target'
SENSITIVITY_DIR = 'sensitivity'
EMULATOR_DIR = 'emulator'
UQ_DIR = 'uq'
#: the input of a step on an emulator trained outside the workflow (emulator_dir)
EMULATOR_INPUT = 'emulator'
#: the input that is a CellML model's file with the files it imports
CELLML_INPUT = 'cellml_model'
#: the samples a uq step's Laplace approximation stores, unless UQ_options.num_samples says
LAPLACE_SAMPLES = 2000
#: where a uq step's posterior predictive check is written, in its step directory
POSTERIOR_PREDICTIVE_DIR = 'posterior_predictive'
#: the draws it simulates, unless UQ_options.posterior_predictive_samples says
POSTERIOR_PREDICTIVE_SAMPLES = 100
#: the error vectors a calibration step writes (param_id/plot_outputs.py), in its run directory
ERROR_VECTOR_FILES = ('percent_error_vec.npy', 'error_vec_names.npy')
#: what a calibration step's result keeps of each held-out item's validation (the series are in
#: its ``validation_results`` file)
VALIDATION_SUMMARY_KEYS = ('data_item_name', 'data_reference', 'operand', 'operation', 'unit',
                           'data_type', 'n_points', 'rmse', 'nrmse', 'mean_abs_z', 'within_2std')
#: where an identifiability step's profile likelihood is written, in its run directory
PROFILE_DIR = 'profile_likelihood'
SOBOL_INDICES_GLOB = 'all_outputs_n*_Sobol_indices.csv'
#: the result keys the study manifest lists as each kind's outputs
STEP_OUTPUTS = {'calibration': ('param_id_output_dir', 'distribution_dir', 'calibrated_model',
                                'percent_error_vec', 'error_vec_names', 'validation_results'),
                'sensitivity': ('sensitivity_dir', 'sobol_indices', 'local_sensitivity_signed'),
                'reduction': ('params_for_id',),
                'emulation': ('emulator_dir',),
                'identifiability': ('laplace', 'profile_likelihood_dir'),
                'set': (),
                'combine': (),
                'sweep': ('sweep_dir', 'sweep_results'),
                'uq': ('uq_output_dir', 'distribution_dir', 'posterior_predictive_dir',
                       'posterior_predictive_draws', 'posterior_predictive_coverage',
                       'posterior_predictive_series'),
                'simulate': ('sweep_dir', 'sweep_results'),
                'figure': ('figure_dir', 'figures')}
#: what a value set by a set step is marked with
SET_BY_HAND = 'set_by_hand'
assert SET_BY_HAND in ORIGINS

# what a step's user_inputs get unless the workflow's settings say otherwise
STEP_DEFAULTS = {'model_type': DEFAULT_MODEL_TYPE, 'DEBUG': False, 'do_ad': DEFAULT_DO_AD, 'do_ia': False,
                 'UQ_options': {}}


class WorkflowStepError(RuntimeError):
    '''A step that failed while running.'''

    def __init__(self, step_id, message):
        super().__init__(f'step "{step_id}": {message}')
        self.step_id = step_id
        self.message = message

    def __reduce__(self):
        # pickled to reach the other ranks (run_on_root)
        return (type(self), (self.step_id, self.message))


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def workflow_digest(path):
    '''The sha256 of a workflow file's content without its figure steps, whatever the key
    order: adding or changing a figure step after a run leaves the run's values current.'''
    with open(path, encoding='utf-8-sig') as f:
        raw = json.load(f)
    if isinstance(raw.get('steps'), list):
        raw['steps'] = [step for step in raw['steps']
                        if not (isinstance(step, dict) and step.get('kind') == 'figure')]
    return settings_digest(raw)


def settings_digest(settings):
    '''The sha256 of a step's settings, whatever their key order.'''
    text = json.dumps(settings, sort_keys=True, default=str)
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def step_settings_digest(workflow, step):
    '''The digest a step's ``settings`` input records: a set step's entries, a combine step's
    inputs and options, a reduction's options, a figure step's declaration, or the workflow's
    settings with the step's on top. Formatting and key order of the file do not change it.'''
    if step.kind == 'figure':
        return settings_digest(figures.declaration(step))
    if step.kind == 'set':
        return settings_digest(step.set_values + step.set_thetas)
    if step.kind == 'combine':
        return settings_digest({'combine_from': step.combine_from, 'combine': step.combine})
    if step.kind == 'reduction':
        return settings_digest(step.reduction)
    return settings_digest(workflow.step_settings(step))


def directory_digest(directory):
    '''The sha256 of the files under ``directory``: their relative paths and contents.'''
    digest = hashlib.sha256()
    for root, dirs, files in os.walk(directory):
        dirs.sort()
        for name in sorted(files):
            path = os.path.join(root, name)
            digest.update(os.path.relpath(path, directory).encode('utf-8'))
            digest.update(sha256(path).encode('ascii'))
    return digest.hexdigest()


def cellml_digest(path):
    '''The sha256 of a CellML file and of every file it imports.'''
    from libcuflynx.calibration_workflow.cellml_files import cellml_imports
    digest = hashlib.sha256(sha256(path).encode('ascii'))
    for imported in cellml_imports(path):
        digest.update(sha256(imported).encode('ascii'))
    return digest.hexdigest()


def input_digest(key, path):
    '''The digest a step's input ``key`` records for ``path``: the CellML model's covers its
    imports, an emulator's its directory, the module library's every library directory, and any
    other input's is the file's sha256. Both the runner and the provenance graph use it.'''
    if key == CELLML_INPUT:
        return cellml_digest(path)
    if key == EMULATOR_INPUT:
        return directory_digest(path)
    if key == LIBRARY_INPUT:
        return module_library_digest(path.split(os.pathsep))
    return sha256(path)


def module_library_digest(library_dirs):
    '''The sha256 of the module files under ``library_dirs``: their paths within each directory
    and their contents.'''
    digest = hashlib.sha256()
    for directory in library_dirs:
        for root, dirs, files in os.walk(directory):
            dirs.sort()
            for name in sorted(files):
                if name.endswith(LIBRARY_FILE_SUFFIXES):
                    path = os.path.join(root, name)
                    digest.update(os.path.relpath(path, directory).encode('utf-8'))
                    digest.update(sha256(path).encode('ascii'))
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


def _libcellml_version():
    try:
        from libcellml import versionString
        return versionString()
    except ImportError:
        return None


def default_output_dir(workflow):
    '''Where a run goes when it is given no output directory:
    ``workflow_output/<workflow_name>`` beside the workflow file (the working directory for a
    workflow given as a dict), which is ``WorkflowBuilder.output_dir``.'''
    base = os.path.dirname(workflow.path) if workflow.path else os.getcwd()
    return os.path.join(base, 'workflow_output', workflow.name)


def _profile_record(summary_path):
    '''What an identifiability step's result keeps of a profile likelihood
    (``ProfileLikelihood._save_results``): its directory, summary and options, and per parameter
    the profile's file, the value profiled about, its cost and the 95% interval. The sweeps
    (``param_values``, ``costs``, ``optima``) are in each file.'''
    with open(summary_path, encoding='utf-8') as f:
        summary = json.load(f)
    directory = os.path.dirname(summary_path)
    profiles = []
    for path in sorted(glob.glob(os.path.join(directory, 'param_*.json')),
                       key=lambda p: int(os.path.basename(p).split('_')[1])):
        with open(path, encoding='utf-8') as f:
            profile = json.load(f)
        name = profile['param_name']
        profiles.append({'parameter': '+'.join(name) if isinstance(name, list) else name,
                         'file': path, 'best_val': profile['best_val'],
                         'best_cost': profile['best_cost'], 'ci_95': profile['ci_95']})
    return {'dir': directory, 'summary': summary_path, 'num_points': summary['num_points'],
            'range_factor': summary['range_factor'],
            'nll_threshold_95': summary['nll_threshold_95'], 'profiles': profiles}


def _member_values(step, param_id, info, values):
    '''``(member_values, modifiers)`` of a calibration's ``values`` (one per params_for_id
    entry): per entry, the value of each parameter it names, and a record of each modifier
    entry. A ``targets`` entry gives its value to every parameter. A ``modifies`` entry's value
    is theta, so its parameters get ``modifier(theta, baseline_i)`` by the engine's own
    expansion, at the baselines ``param_id`` (the run's CVS0DParamID) resolved; the record is
    ``{name, modifier, targets, theta, baselines, values}``.'''
    from libcuflynx.parsers.PrimitiveParsers import expand_modifier_param_vals
    names = [list(q) if isinstance(q, (list, tuple)) else [q] for q in info['param_names']]
    if not info.get('modifiers'):
        return [[float(value)] * len(q) for q, value in zip(names, values)], []
    resolved = getattr(getattr(param_id, 'param_id', None), 'param_id_info', None)
    if not resolved or len(resolved.get('param_names', ())) != len(names):
        raise WorkflowStepError(step.id, 'its params_for_id has modifier entries, and the '
                                         'calibration kept no baselines to give their '
                                         'parameters values from.')
    expanded = expand_modifier_param_vals(resolved, [float(v) for v in values])
    by_index = {mod['index']: mod for mod in resolved['modifiers']}
    member_values, modifiers = [], []
    for index, (q, value) in enumerate(zip(names, expanded)):
        if index not in by_index:
            member_values.append([float(value)] * len(q))
            continue
        mod = by_index[index]
        member_values.append([float(v) for v in value])
        modifiers.append({'name': mod['name'], 'modifier': mod.get('modifier'),
                          'targets': list(mod['targets']), 'theta': float(values[index]),
                          'baselines': [float(b) for b in mod['baselines']],
                          'values': member_values[-1]})
    return member_values, modifiers


def _write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def _read_json(path):
    with open(path) as f:
        return json.load(f)


def read_step_result(path):
    '''The step_result.json at ``path``. One whose records name their module ``vessel`` is
    refused, naming the upgrade that renames it.'''
    from libcuflynx.upgrade import hint, is_older_step_result
    result = _read_json(path)
    if is_older_step_result(result):
        run_dir = os.path.dirname(os.path.dirname(os.path.abspath(path)))
        raise WorkflowError(f'{path} names the module of each value "vessel" (now "module"). '
                            + hint('workflow', run_dir))
    return result


def given_values(result):
    '''The records of what a step's result passes on: what it calibrated, combined or set by
    hand.'''
    return result['calibrated'] + result.get('set', [])


def _from_step_hint(step_id):
    '''How to run from a step, for a Python caller and for the command line.'''
    return f'wf.run(from_step={step_id!r}) in Python, --from-step {step_id} on the command line'


def target_values(step, result):
    '''The records of what a step's result gives the target: none for a point step, whose values
    are only for ``start_from``, ``result_from`` or ``fixed_from``.'''
    return [] if step.point else given_values(result)


def _how_set(record):
    '''The ``origin`` and ``reason`` of a value set by hand; empty for a calibrated one.'''
    return {key: record[key] for key in ('origin', 'reason') if record.get(key)}


class _Run:
    '''One run of a resolved workflow.'''

    def __init__(self, resolved, output_dir, comm, progress):
        self.resolved = resolved
        self.workflow = resolved.workflow
        self.output_dir = os.path.abspath(output_dir)
        self.comm = comm
        self.rank = comm.Get_rank()
        self.progress = progress
        self._library_digest = None
        #: rank 0: ``{step id: {model name: row}}`` of the values a step's own parameters start
        #: at (start_from), set by _prepare
        self.start_points = {}
        #: per step, the thetas its modifier entries start at or are analysed at
        self.point_thetas = {}
        #: per step on an emulator_dir, {model name: value} of the point it analyses
        self.emulator_points = {}
        #: 'error' or 'warn': what the merge does when unrelated steps give the target one
        #: parameter (resolve.check_merged_once)
        self.target_overlap = 'warn'
        #: rank 0: {step id: what its protocol factors multiplied}, for its step_result
        self._factors = {}

    # -- helpers -------------------------------------------------------------------------

    def emit(self, event, **fields):
        if self.progress is not None and self.rank == 0:
            self.progress(dict(event=event, workflow=self.workflow.name, **fields))

    def on_root(self, fn):
        '''``fn()`` on rank 0, its result (or the exception it raised) on every rank.'''
        return run_on_root(self.comm, fn)

    def step_dir(self, step_id):
        return os.path.join(self.output_dir, step_id)

    def obs_data_read(self, step):
        '''The obs_data a step reads: its own, or the copy written into its directory.'''
        resolved = self.resolved.steps[step.id]
        if resolved.reads_step_inputs:
            return step_input_paths(self.step_dir(step.id), step)[0]
        return resolved.obs_data_path

    def step_result(self, step_id):
        path = os.path.join(self.step_dir(step_id), STEP_RESULT)
        if not os.path.isfile(path):
            return None
        return read_step_result(path)

    def library_input(self):
        '''The ``module_library`` entry of a step's inputs, read once a run.'''
        dirs = self.workflow.module_library_dirs
        if not dirs:
            return {}
        if self._library_digest is None:
            self._library_digest = module_library_digest(dirs)
        return {LIBRARY_INPUT: {'path': os.pathsep.join(dirs), 'sha256': self._library_digest}}

    def inputs(self, step_id):
        rs = self.resolved.steps[step_id]
        if rs.model is None:
            return self._model_free_inputs(rs.step)
        settings = {'path': self.workflow.path,
                    'sha256': step_settings_digest(self.workflow, rs.step)}
        files = {'obs_data': rs.obs_data_path,
                 'params_for_id': self.params_for_id_of(rs.step)}
        if not files['obs_data']:
            del files['obs_data']         # a sweep without one
        if not files['params_for_id'] or not os.path.isfile(files['params_for_id']):
            del files['params_for_id']    # a sweep's, or a reduction's not yet written
        if rs.model.parameters_path and os.path.isfile(rs.model.parameters_path):
            files['parameters'] = rs.model.parameters_path
        if rs.model.kind == 'external':
            files['external_model'] = rs.model.model.path
        if rs.model.kind == 'array':
            files['module_array'] = rs.model.model.module_array
        if rs.selection is not None:
            # a selection starts from the target's values
            files['target_parameters'] = self.workflow.target.parameters
            files['target_module_array'] = self.workflow.target.module_array
        # the user function files the step runs with (a result recorded without them is not
        # stale for that: changed_inputs compares only the keys a result has)
        merged = self.workflow.step_settings(rs.step)
        for key, setting in FUNCS_INPUTS.items():
            if merged.get(setting) and os.path.isfile(merged[setting]):
                files[key] = merged[setting]
        inputs = {key: {'path': path, 'sha256': sha256(path)} for key, path in files.items()}
        if rs.model.kind == 'cellml':
            inputs[CELLML_INPUT] = {'path': rs.model.model.path,
                                    'sha256': input_digest(CELLML_INPUT, rs.model.model.path)}
        if rs.step.emulator_dir and os.path.isdir(rs.step.emulator_dir):
            inputs[EMULATOR_INPUT] = {'path': rs.step.emulator_dir,
                                      'sha256': input_digest(EMULATOR_INPUT,
                                                             rs.step.emulator_dir)}
        return {**inputs, SETTINGS_INPUT: settings, **self.library_input()}

    def _model_free_inputs(self, step):
        '''The inputs of a reduction (its options; the rest are its sensitivity step's, or
        the params_for_id a choice made by hand keeps entries of), a combine step (its inputs
        and options), a set step (its entries, and the target's values when it scales them) or
        a figure step (its declaration, its script and the files it lists).'''
        inputs = {SETTINGS_INPUT: {'path': self.workflow.path,
                                   'sha256': step_settings_digest(self.workflow, step)}}
        if step.chosen_by_hand and os.path.isfile(step.params_for_id):
            inputs['params_for_id'] = {'path': step.params_for_id,
                                       'sha256': sha256(step.params_for_id)}
        if step.kind == 'figure':
            inputs.update({key: {'path': path, 'sha256': sha256(path)}
                           for key, path in figures.input_files(step).items()})
        if any(entry.get('of') == INHERITED_BASE for entry in step.set_values):
            # an external target's values are its model file's defaults, a CellML target's its
            # file's with its parameters file's over them
            target = self.resolved.target
            path = target.model.path if target.kind in ('external', 'cellml') else \
                target.parameters_path
            if target.kind == 'cellml' and target.parameters_path:
                inputs['target_parameters_file'] = {'path': target.parameters_path,
                                                    'sha256': sha256(target.parameters_path)}
            if os.path.isfile(path):
                inputs['target_parameters'] = {'path': path, 'sha256': sha256(path)}
            inputs.update(self.library_input())
        return inputs

    def changed_inputs(self, step_id, result):
        '''The keys of the inputs in ``result`` that differ from the step's inputs now.'''
        current = self.inputs(step_id)
        return [key for key, info in result.get('inputs', {}).items()
                if key in current and current[key]['sha256'] != info.get('sha256')]

    def require_current(self, step_id, needed_by):
        '''The result of ``step_id``, which ``needed_by`` takes something from. A WorkflowError
        when it has not run, when its inputs changed since it ran, or when any of that holds
        for a step it took something from, or such a step ran again after it. A change in the
        module library is a warning: a library holds far more than the model uses.'''
        result = self.step_result(step_id)
        if result is None or result.get('status') != 'done':
            raise WorkflowError(
                f'step "{needed_by}" needs the result of "{step_id}", which has not been run '
                f'into {self.output_dir}. Run the workflow from "{step_id}": '
                f'{_from_step_hint(step_id)}, or run every step.')
        for key in self.changed_inputs(step_id, result):
            if key == LIBRARY_INPUT:
                warnings.warn(f'the module library has changed since "{step_id}" ran; its result '
                              f'may no longer match it.', UserWarning, stacklevel=2)
                continue
            changed = 'its settings' if key == SETTINGS_INPUT else result['inputs'][key]['path']
            raise WorkflowError(
                f'step "{needed_by}" needs the result of "{step_id}", but {changed} has changed '
                f'since "{step_id}" ran. Run the workflow from "{step_id}": '
                f'{_from_step_hint(step_id)}.')
        for source in self.workflow.step(step_id).depends_on:
            upstream = self.require_current(source, needed_by)
            if (upstream.get('finished') or '') > (result.get('started') or ''):
                raise WorkflowError(
                    f'step "{needed_by}" needs the result of "{step_id}", but "{source}", which '
                    f'it took something from, ran again after it. Run the workflow from '
                    f'"{step_id}": {_from_step_hint(step_id)}.')
        return result

    # -- what a step gets from earlier steps ---------------------------------------------

    def fixed_values(self, step, sources, own_only=False, subsets=None):
        '''``{model name in this step: row}`` from every step in ``sources``: its calibrated
        values, those it set by hand and, unless ``own_only``, the values it was itself given
        as fixed. A source in ``subsets`` (``{step id: its model names}``) gives only those of
        its calibrated and set values. A value given by two steps is the later one's
        (:meth:`later_value`).'''
        subsets = subsets or {}
        values = {}
        for source in sources:
            result = self.require_current(source, step.id)
            vessel_of = self.resolved.map_between(source, step.id)
            left_out = []
            records = given_values(result) + ([] if own_only else result['fixed'])
            if source in subsets:
                records = [r for r in given_values(result)
                           if r['model_name'] in subsets[source]]
            for record in records:
                vessel = vessel_of(record['module'])
                if vessel is None:
                    left_out.append(record['model_name'])
                    continue
                name = naming.model_name(vessel, record['param'])
                origin = record.get('from_step', source)
                row = {'module': vessel,
                       'param': record['param'], 'model_name': name,
                       'host_name': naming.host_row_name(vessel, record['param']),
                       'value': record['value'], 'units': record.get('units', ''),
                       'from_step': origin, **_how_set(record)}
                values[name] = self.later_value(step, values.get(name), row)
            if left_out:
                model = step.model.label() if step.model else self.workflow.target.label()
                warnings.warn(f'step "{step.id}": {len(left_out)} value(s) of "{source}" are '
                              f'not fixed, because {model} has no such vessel: '
                              f'{sorted(left_out)}.', UserWarning, stacklevel=2)
        return values

    def later_value(self, step, old, new):
        '''Which of two rows giving ``step`` one parameter is used (``old`` may be None): that
        of the step that takes from the other's step, directly or through other steps; else,
        when one is a set step's, that of the later step in run order. Two different values
        from other unrelated steps are a WorkflowError (resolve.given_twice).'''
        if old is None or old['from_step'] == new['from_step']:
            return new
        first, second = old['from_step'], new['from_step']
        if not hasattr(self, '_ancestors'):
            self._ancestors = self.workflow.ancestors()
            self._run_order = [s.id for s in self.workflow.topological_order()]
        if first in self._ancestors[second]:
            return new
        if second in self._ancestors[first]:
            return old
        if 'set' not in (self.workflow.step(first).kind, self.workflow.step(second).kind) and \
                not np.isclose(old['value'], new['value'], rtol=1e-12, atol=0.0):
            raise WorkflowError(
                f'step "{step.id}" is given two different values for "{new["model_name"]}": '
                f'{old["value"]} (from "{first}") and {new["value"]} (from "{second}"), and '
                f'neither step takes from the other. Add a combine step that gives one value '
                f'from them ("kind": "combine", "combine_from": {json.dumps([first, second])}) '
                f'and take from that.')
        return new if self._run_order.index(second) > self._run_order.index(first) else old

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
               'identifiability': self._run_identifiability, 'set': self._run_set,
               'combine': self._run_combine,
               'sweep': self._run_sweep, 'uq': self._run_uq, 'simulate': self._run_sweep,
               'figure': self._run_figure}[step.kind]
        record = run(step, self.workflow.step_settings(step))

        def finish():
            result = dict({'step': step.id, 'kind': step.kind, 'status': 'done',
                           'model': step.model.as_dict() if step.model is not None else None,
                           'place': self.resolved.steps[step.id].place if step.model else '',
                           'started': started, 'finished': _now(),
                           'duration_s': round(time.time() - t0, 3),
                           'calibrated': [], 'fixed': [], 'inputs': self.inputs(step.id)},
                          **record)
            factors = self._factors.pop(step.id, None)
            if factors:
                result['protocol_factors'] = factors
            _write_json(os.path.join(self.step_dir(step.id), STEP_RESULT), result)
            return result

        result = self.on_root(finish)
        self.emit('step_finished', step=step.id, index=index, total=total, kind=step.kind,
                  best_cost=result.get('best_cost'), duration_s=result['duration_s'])
        return result

    def params_for_id_of(self, step):
        '''The params_for_id a step runs with: its own, or the one its reduction wrote. A sweep
        has none.'''
        if step.params_for_id_from:
            return reduced_params_for_id_path(self.output_dir, step.params_for_id_from)
        return step.params_for_id

    def own_parameters(self, step, params_for_id):
        '''The model names, in the step's model, of what ``params_for_id`` (the step's)
        calibrates or analyses.'''
        resolved = self.resolved.steps[step.id]
        own = read_calibrated_parameters(params_for_id, step.id) if params_for_id else []
        own = [resolved.in_model(p) for p in own]
        if resolved.renames_inputs:
            own = [CalibratedParameter(resolved.vessel_map.from_target(p.vessel) or p.vessel,
                                       p.param) for p in own]
        return {p.model_name for p in own}

    def own_modifiers(self, step, params_for_id):
        '''``{entry name: the model names, in the step's model, of what it modifies}`` of the
        modifier entries of ``params_for_id`` (the step's).'''
        if not params_for_id:
            return {}
        from libcuflynx.parsers.PrimitiveParsers import ObsAndParamDataParser
        resolved = self.resolved.steps[step.id]
        info = ObsAndParamDataParser(
            modifier_funcs_external_path=self.workflow.step_settings(step).get(
                'modifier_funcs_external_path')).get_param_id_info(params_for_id)
        modifiers = {}
        for mod in info.get('modifiers') or []:
            names = set()
            for qname in mod['targets']:
                vessel, _, param = str(qname).partition('/')
                if resolved.renames_inputs:
                    vessel = resolved.vessel_map.from_target(vessel) or vessel
                names.add(resolved.in_model(
                    CalibratedParameter(vessel=vessel, param=param)).model_name)
            modifiers[mod['name']] = names
        return modifiers

    def thetas_of(self, step, source, modifiers):
        '''``{entry name: theta}`` that ``source``'s result gives the modifier entries
        ``modifiers`` of ``step``: a set step's theta entries, or the thetas a calibration
        found. A set step's theta for an entry the step does not have is an error.'''
        result = self.require_current(source, step.id)
        declared = {entry['entry']: entry['theta'] for entry in result.get('thetas') or []}
        unknown = sorted(set(declared) - set(modifiers))
        if unknown:
            raise WorkflowError(f'step "{step.id}" is given the theta of {unknown} by '
                                f'"{source}", and its params_for_id has no modifier entry of '
                                f'that name (it has {sorted(modifiers)}).')
        found = {mod['name']: mod['theta'] for mod in result.get('modifiers') or []}
        found.update(declared)
        return {name: float(theta) for name, theta in found.items() if name in modifiers}

    def _prepare(self, step, settings):
        '''Rank 0: a clean step directory and the step's model, generated with the values it is
        given (``fixed_from``, and ``result_from`` including the parameters it analyses), or
        copied from its ``emulator_from`` step. Returns ``(generated, params_for_id, units,
        fixed, priors)``; ``fixed`` leaves out the parameters the step analyses.'''
        resolved = self.resolved.steps[step.id]
        step_dir = self.step_dir(step.id)
        params_for_id = self.params_for_id_of(step)
        if step.params_for_id_from:
            self.require_current(step.params_for_id_from, step.id)
        own = self.own_parameters(step, params_for_id)
        if step.on_target:
            # the values merge() gives the target: each source's own values, in run order, a
            # later step winning
            fixed = {}
            for source in self.workflow.topological_order():
                if source.id in step.fixed_from:
                    fixed.update(self.fixed_values(step, [source.id], own_only=True))
        else:
            # a parameter the step calibrates itself is not fixed: the later calibration wins
            fixed = {n: row for n, row in
                     self.fixed_values(step, step.fixed_from,
                                       subsets=step.fixed_parameters).items() if n not in own}
        point = {}
        if step.result_from:
            # result_from's values of the step's own parameters are the point it analyses (its
            # result records them as ``at`` or ``nominal``); the rest are fixed
            for name, row in self.fixed_values(step, [step.result_from]).items():
                if name in own:
                    point[name] = row
                else:
                    fixed[name] = self.later_value(step, fixed.get(name), row)
        start = {}
        if step.start_from:
            # its own parameters start at the values of start_from; they are not fixed
            start = {n: row for n, row in self.fixed_values(step, [step.start_from],
                                                             own_only=True).items() if n in own}
        # A modifier entry's point is its theta, about the values its parameters have in the
        # step's model: where the source gives that theta, the parameters keep those values.
        thetas = {}
        modifiers = self.own_modifiers(step, params_for_id)
        for source, values in ((step.result_from, point), (step.start_from, start)):
            if not source:
                continue
            for name, theta in self.thetas_of(step, source, modifiers).items():
                thetas[name] = theta
                for model_name in modifiers[name]:
                    values.pop(model_name, None)
        self.point_thetas[step.id] = thetas
        if step.start_from:
            if not start and not thetas:
                raise WorkflowError(f'step "{step.id}" starts from "{step.start_from}", which '
                                    f'has no value of any parameter it calibrates.')
        self.start_points[step.id] = start
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
                                            f'{self.workflow.name}, step {row["from_step"]}',
                          **_how_set(row)}
                         for name, row in sorted(fixed.items())]
            overrides += [{'variable_name': row['host_name'], 'units': row['units'],
                           'value': repr(float(row['value'])),
                           'data_reference': f'the point analysed: calibration workflow '
                                             f'{self.workflow.name}, step {row["from_step"]}',
                           **_how_set(row)}
                          # An emulator answers for the model it was trained on: the point of a
                          # step on one is only where it is evaluated, and stays out of the model.
                          for name, row in sorted(point.items()) if not step.emulator_dir]
            overrides += [{'variable_name': row['host_name'], 'units': row['units'],
                           'value': repr(float(row['value'])),
                           'data_reference': f'start of the fit: calibration workflow '
                                             f'{self.workflow.name}, step {row["from_step"]}'}
                          for name, row in sorted(start.items())]
            generated = generate_step_model(
                resolved, os.path.join(step_dir, 'model'), step.id,
                under(resolved.base_rows, overrides), obs_data, self.resolved.library_inputs,
                settings)
        self.emulator_points[step.id] = ({name: float(row['value'])
                                          for name, row in point.items()}
                                         if step.emulator_dir else {})
        model_rows = {r['variable_name']: r
                      for r in read_parameter_rows(generated['parameters_path'])}
        for name, row in {**fixed, **point}.items():
            if name not in model_rows:
                raise WorkflowError(
                    f'step "{step.id}": the value fixed from "{row["from_step"]}" for '
                    f'"{name}" has no parameter of that name in the step model '
                    f'({generated["parameters_path"]}).')
        if has_factors(obs_data):
            # against the values the step's model was generated with
            refused = {n: 'calibrates' for n in own}
            if step.kind == 'sweep':
                refused.update({p.model_name: 'sweeps' for p in resolved.parameters})
            resolved_obs = self.obs_data_read(step)
            os.makedirs(os.path.dirname(resolved_obs), exist_ok=True)
            self._factors[step.id] = resolve_factors(
                obs_data, resolved_obs, model_rows, fixed, refused, f'step "{step.id}"')
        if priors:
            source_path = params_for_id
            params_for_id = os.path.join(step_dir, f'{step.id}_params_for_id.json')
            _write_params_for_id_with_priors(source_path, params_for_id, priors, step.id)
        if thetas:
            source_path = params_for_id
            params_for_id = os.path.join(step_dir, f'{step.id}_params_for_id.json')
            _write_params_for_id_with_thetas(source_path, params_for_id, thetas, step.id)
        units = {k: v.get('units', '') for k, v in model_rows.items()}
        return generated, params_for_id, units, fixed, priors

    def _inp(self, step, settings, generated, params_for_id):
        '''The user_inputs of one step's run. A step on an emulator runs the emulation step's
        model (its files carry that step's prefix) with its emulator.'''
        step_dir = self.step_dir(step.id)
        prefix = step.emulator_from or step.id
        resolved = self.resolved.steps[step.id]
        obs_data = self.obs_data_read(step)
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
            # the step's manifest copies its inputs into the workflow's working dir
            'cuflynx_working_dir': self.output_dir,
            'do_uq': step.store_distribution,
        })
        inp.update({k: v for k, v in self.resolved.library_inputs.items()})
        if resolved.model.kind == 'external':
            inp.update(external_inputs(settings, generated))
        if step.emulator_from:
            emulation = self.workflow.step(step.emulator_from)
            inp['use_emulator'] = True
            inp['emulator_settings'] = dict(
                self.workflow.step_settings(emulation).get('emulator_settings') or {},
                emulator_dir=emulator_dir(self.output_dir, emulation.id))
        if step.emulator_dir:
            inp['use_emulator'] = True
            inp['emulator_settings'] = dict(settings.get('emulator_settings') or {},
                                            emulator_dir=step.emulator_dir)
        return inp

    def _run_stage(self, step, what, fn, *args):
        try:
            return fn(*args)
        except CAInputError as exc:
            raise WorkflowStepError(step.id, f'the {what} configuration was refused: '
                                             f'{exc}') from exc
        except SystemExit as exc:
            # library code that still exit()s on bad input, after printing why (#706)
            raise WorkflowStepError(step.id, f'the {what} configuration was refused (exit '
                                             f'{exc.code}); the messages above say why.')

    def _run_calibration(self, step, settings):
        from libcuflynx.scripts.param_id_run_script import run_param_id

        resolved = self.resolved.steps[step.id]
        generated, params_for_id, units, fixed, priors = self.on_root(
            lambda: self._prepare(step, settings))
        # A reduction's parameters on the emulator of all of them: the run is on the
        # emulator's params_for_id, calibrating the step's entries and holding the others.
        run_params_for_id, kept = params_for_id, None
        if self.workflow.holds_entries(step):
            run_params_for_id, kept = self.on_root(
                lambda: self._emulator_entries(step, params_for_id))
        inp = self._inp(step, settings, generated, run_params_for_id)
        if kept is not None:
            inp['calibrated_entries'] = kept
        if priors:
            inp['joint_priors'] = [([n for n in dist.names],
                                    dist.as_prior(prior.kind, prior.inflate))
                                   for prior, dist in priors]
        outcome = self._run_stage(step, 'param_id', run_param_id, inp)
        outputs = self.on_root(lambda: self._calibration_outputs(step, outcome, generated))

        def finish():
            case_dir = outcome['output_dir']
            from libcuflynx.parsers.PrimitiveParsers import (ObsAndParamDataParser,
                                                             param_entry_labels)
            info = ObsAndParamDataParser().get_param_id_info(run_params_for_id)
            labels = param_entry_labels(info)
            best = np.atleast_1d(np.load(os.path.join(case_dir, 'best_param_vals.npy')))
            best_cost = float(np.load(os.path.join(case_dir, 'best_cost.npy')))
            distribution_dir = None
            if step.store_distribution:
                distribution_dir = self._store_distribution(
                    step, outcome['uq_output_dir'], info, settings)
            values, point_estimate = best, POINT_BEST_FIT
            if priors and not uses_log_posterior(settings):
                # The optimiser minimised the cost, which knows nothing of the priors; only the
                # MCMC sampled the posterior they shape. Report that posterior's medians.
                samples = StoredDistribution.load(distribution_dir).samples
                values, point_estimate = np.median(samples, axis=0), POINT_POSTERIOR_MEDIAN
            calibrated, held = [], []
            # an entry's value is each of its parameters' value, except a modifier's, whose
            # value is theta: its parameters have its function of theta at their baselines
            member_values, modifiers = _member_values(step, outcome.get('param_id'), info,
                                                      values)
            for label, qnames, entry_values in zip(labels, info['param_names'], member_values):
                # an entry the step does not calibrate was held at this value
                rows = calibrated if kept is None or label in kept else held
                names = qnames if isinstance(qnames, (list, tuple)) else [qnames]
                for qname, value in zip(names, entry_values):
                    vessel, _, param = str(qname).partition('/')
                    # as the step's model names it (a CellML's constant)
                    parameter = resolved.in_model(CalibratedParameter(vessel=vessel, param=param))
                    vessel, param = parameter.vessel, parameter.param
                    rows.append({
                        'module': vessel, 'param': param, 'model_name': parameter.model_name,
                        'target_model_name': resolved.target_model_name(parameter),
                        'target_host_name': naming.host_row_name(resolved.to_target(vessel),
                                                                 param),
                        'target_parameterisation_name':
                            resolved.target_parameterisation_name(parameter),
                        'value': float(value), 'units': units.get(parameter.model_name, '')})
            return {
                'method': settings.get('param_id_method'), 'best_cost': best_cost,
                'point_estimate': point_estimate,
                # the calibrated entries' values, in the order of params_for_id_used
                'best_fit': [float(v) for label, v in zip(labels, best)
                             if kept is None or label in kept],
                'calibrated': calibrated,
                'modifiers': modifiers,
                # (a step on an emulator of more parameters than it calibrates) the emulator's
                # other parameters, held at their values in the emulation step's model
                **({'held': held, 'held_at': f'the model of "{step.emulator_from}"',
                    'emulator_params_for_id': run_params_for_id} if kept is not None else {}),
                'fixed': [dict(row) for _, row in sorted(fixed.items())],
                'priors': [dict(prior.as_dict(), parameters=dist.names)
                           for prior, dist in priors],
                'params_for_id_used': params_for_id,
                'settings': settings,
                # what the run used: the settings' solver_info with the parser's defaults
                'solver_info': outcome['solver_info'],
                'generated': generated,
                'model_path': generated['model_path'],
                'param_id_output_dir': case_dir,
                'distribution_dir': distribution_dir,
                **outputs,
                **self._start_record(step),
            }

        return self.on_root(finish)

    def _emulator_entries(self, step, params_for_id):
        '''Rank 0, for a step that calibrates some of its emulator's parameters: the
        params_for_id the emulator was trained on, and the labels of the entries of
        ``params_for_id`` (the step's), which must all be entries of it.'''
        from libcuflynx.parsers.PrimitiveParsers import (ObsAndParamDataParser,
                                                         param_entry_labels)
        trained_on = self.require_current(step.emulator_from, step.id)['params_for_id_used']
        parser = ObsAndParamDataParser(
            modifier_funcs_external_path=self.workflow.step_settings(step).get(
                'modifier_funcs_external_path'))
        kept = param_entry_labels(parser.get_param_id_info(params_for_id))
        known = param_entry_labels(parser.get_param_id_info(trained_on))
        unknown = [label for label in kept if label not in known]
        if unknown:
            raise WorkflowStepError(
                step.id, f'it calibrates {unknown}, which the emulator of '
                         f'"{step.emulator_from}" was not trained on (its parameters are '
                         f'{known}).')
        return trained_on, kept

    def _calibration_outputs(self, step, outcome, generated):
        '''Rank 0: what a calibration step writes after its fit, with CA's own writers -- the
        observables at the best fit and their error vectors (``plot_outputs``, in the run
        directory), the best fit against the obs_data's held-out prediction items
        (``save_prediction_data``: ``validation_results.json`` in the run directory, and each
        item's statistics under ``validation``), and, when its model has a CellML, that model
        with the best fit (``<step>_calibrated.cellml`` in the step directory). A failure is a
        warning: the fit stands without them.'''
        param_id = outcome.get('param_id')
        out = {'percent_error_vec': None, 'error_vec_names': None, 'calibrated_model': None,
               'validation_results': None, 'validation': []}
        if param_id is None:
            return out
        case_dir = outcome['output_dir']
        try:
            param_id.simulate_with_best_param_vals()
            param_id.plot_outputs()
            for name in ERROR_VECTOR_FILES:
                path = os.path.join(case_dir, name)
                out[name[:-len('.npy')]] = path if os.path.isfile(path) else None
        except (Exception, SystemExit) as exc:
            # plot_outputs exits on a label it cannot draw
            warnings.warn(f'step "{step.id}": the observables at its best fit were not written: '
                          f'{exc}', UserWarning, stacklevel=2)
        try:
            from libcuflynx.param_id.validation import VALIDATION_RESULTS_FILE
            param_id.save_prediction_data()
            path = os.path.join(case_dir, VALIDATION_RESULTS_FILE)
            if os.path.isfile(path):
                with open(path, encoding='utf-8') as f:
                    items = json.load(f)['items']
                out['validation_results'] = path
                out['validation'] = [{key: item.get(key) for key in VALIDATION_SUMMARY_KEYS}
                                     for item in items]
        except (Exception, SystemExit) as exc:
            warnings.warn(f'step "{step.id}": its best fit was not validated against the '
                          f'held-out data: {exc}', UserWarning, stacklevel=2)
        cellml = generated.get('flat_model_path') or (
            generated['model_path'] if str(generated['model_path']).endswith('.cellml')
            else None)
        if cellml:
            try:
                out['calibrated_model'] = param_id.write_calibrated_model(
                    cellml_path=cellml, output_dir=self.step_dir(step.id))
            except Exception as exc:
                warnings.warn(f'step "{step.id}": its calibrated model was not written: {exc}',
                              UserWarning, stacklevel=2)
        return out

    def _start_record(self, step):
        '''The ``start_from`` record of a step's result: the step and the values its own
        parameters started at.'''
        if not step.start_from:
            return {}
        thetas = self.point_thetas.get(step.id)
        return {'start_from': {'step': step.start_from,
                               'values': [dict(row) for _, row in
                                          sorted(self.start_points.get(step.id, {}).items())],
                               # the modifier entries that start at a theta of start_from
                               **({'thetas': dict(thetas)} if thetas else {})}}

    def _run_sensitivity(self, step, settings):
        from libcuflynx.scripts.sensitivity_analysis_run_script import run_SA

        generated, params_for_id, _, fixed, _ = self.on_root(
            lambda: self._prepare(step, settings))
        out_dir = os.path.join(self.step_dir(step.id), SENSITIVITY_DIR)
        sa_options = dict(settings['sa_options'], output_dir=out_dir)
        inp = self._inp(step, settings, generated, params_for_id)
        inp['sa_options'] = sa_options
        outcome = self._run_stage(step, 'sensitivity analysis', run_SA, inp)

        def finish():
            method = sa_options.get('method', 'sobol')
            found = sorted(glob.glob(os.path.join(out_dir, SOBOL_INDICES_GLOB)))
            if method == 'sobol' and not found:
                raise WorkflowStepError(step.id, f'the Sobol analysis wrote no indices into '
                                                 f'{out_dir}.')
            nominal, signed = {}, None
            if method == 'local':
                from libcuflynx.sensitivity_analysis.sensitivityAnalysis import LOCAL_SIGNED_FILE
                # the analysis is about the model's values: result_from's, or its own
                names, values = self._model_point(step, generated, params_for_id)
                nominal = {'nominal': {'source': step.result_from or step.emulator_from or
                                       'model', 'parameters': names, 'values': values}}
                signed = os.path.join(out_dir, LOCAL_SIGNED_FILE)
                if not os.path.isfile(signed):
                    raise WorkflowStepError(step.id, f'the local analysis wrote no {signed}.')
            return {'method': method, 'sensitivity_dir': out_dir, **nominal,
                    'sobol_indices': found[0] if found else None,
                    'local_sensitivity_signed': signed,
                    'fixed': [dict(row) for _, row in sorted(fixed.items())],
                    'params_for_id_used': params_for_id, 'settings': settings,
                    'solver_info': outcome['solver_info'],
                    'generated': generated, 'model_path': generated['model_path']}

        return self.on_root(finish)

    def _run_reduction(self, step, settings):
        def reduce():
            reduction = ParamReduction.from_options(step.reduction)
            if step.chosen_by_hand:
                source = {'sensitivity_dir': None, 'params_for_id_used': step.params_for_id}
                what = f'keeping entries of {step.params_for_id}'
            else:
                source = self.require_current(step.sensitivity_from, step.id)
                what = f'reducing "{step.sensitivity_from}"'
            step_dir = self.step_dir(step.id)
            if os.path.isdir(step_dir):
                shutil.rmtree(step_dir)
            os.makedirs(step_dir)
            out = reduced_params_for_id_path(self.output_dir, step.id)
            try:
                selected = reduction.reduce(source['sensitivity_dir'],
                                            source['params_for_id_used'], out)
            except ValueError as exc:
                raise WorkflowStepError(step.id, f'{what}: {exc}') from exc
            return {'sensitivity_step': step.sensitivity_from,
                    'reduction': reduction.as_dict(), 'selected': selected,
                    'kept_rows': len(selected), 'params_for_id': out,
                    'report': os.path.join(step_dir, REDUCTION_REPORT_FILE)}

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
                    # the user_inputs parser fills in the dict it is given
                    'solver_info': inp['solver_info'],
                    'generated': generated, 'model_path': generated['model_path']}

        return self.on_root(finish)

    def _run_identifiability(self, step, settings):
        from libcuflynx.scripts.identifiability_run_script import run_identifiability_analysis
        from libcuflynx.parsers.PrimitiveParsers import ObsAndParamDataParser

        generated, params_for_id, units, fixed, _ = self.on_root(
            lambda: self._prepare(step, settings))
        info = ObsAndParamDataParser().get_param_id_info(params_for_id)
        # the analysis is at the result_from step's values, which the model was generated with,
        # or at the model's own
        _, best = self.on_root(lambda: self._model_point(step, generated, params_for_id))
        inp = self._inp(step, settings, generated, params_for_id)
        inp['ia_options'] = settings['ia_options']
        self._run_stage(step, 'identifiability analysis', run_identifiability_analysis, inp,
                        best)

        def finish():
            record = {'method': settings['ia_options'].get('method'),
                      'parameters': list(info.get('param_labels') or []),
                      'at': [float(v) for v in best], 'at_source': step.result_from or 'model',
                      'fixed': [dict(row) for _, row in sorted(fixed.items())],
                      'params_for_id_used': params_for_id, 'settings': settings,
                      # the user_inputs parser fills in the dict it is given
                      'solver_info': inp['solver_info'],
                      'generated': generated, 'model_path': generated['model_path']}
            prefix = os.path.join(self.step_dir(step.id), step.emulator_from or step.id)
            mean, covariance = prefix + '_laplace_mean.npy', prefix + '_laplace_covariance.npy'
            if os.path.isfile(covariance):
                cov = np.atleast_2d(np.load(covariance))
                record['laplace'] = {'mean': np.atleast_1d(np.load(mean)).tolist(),
                                     'covariance': cov.tolist(),
                                     'std': np.sqrt(np.diag(cov)).tolist(),
                                     'mean_path': mean, 'covariance_path': covariance}
            summaries = glob.glob(os.path.join(inp['param_id_output_dir'], PROFILE_DIR,
                                               'summary.json'))
            if summaries:
                record['profile_likelihood'] = _profile_record(summaries[0])
                record['profile_likelihood_dir'] = record['profile_likelihood']['dir']
            return record

        return self.on_root(finish)

    def _model_point(self, step, generated, params_for_id):
        '''Rank 0: ``(names, values)``: each params_for_id entry (its first member's
        ``vessel/param``) and its value in the step's generated model, or, for a step on an
        ``emulator_dir``, at its ``result_from`` point (which its model is not given). A
        modifier entry's value is its theta: the entry's ``theta`` (written from the step's
        ``result_from``), else the one that leaves its parameters at the model's values.'''
        from libcuflynx.parsers.PrimitiveParsers import PARAM_MODIFIERS, ObsAndParamDataParser
        info = ObsAndParamDataParser().get_param_id_info(params_for_id)
        rows = {r['variable_name']: float(r['value'])
                for r in read_parameter_rows(generated['parameters_path'])}
        rows.update(self.emulator_points.get(step.id, {}))
        modifiers = {mod['index']: mod for mod in info.get('modifiers') or []}
        names, values = [], []
        for index, qnames in enumerate(info['param_names']):
            first = str(qnames[0] if isinstance(qnames, (list, tuple)) else qnames)
            vessel, _, param = first.partition('/')
            names.append(first)
            if index not in modifiers:
                values.append(rows[self.resolved.steps[step.id].in_model(
                    CalibratedParameter(vessel=vessel, param=param)).model_name])
                continue
            mod = modifiers[index]
            theta = mod.get('theta')
            if theta is None:
                theta = (PARAM_MODIFIERS.get(mod.get('modifier')) or {}).get('identity')
            if theta is None:
                raise WorkflowError(
                    f'{params_for_id}: the modifier entry "{mod["name"]}" '
                    f'({mod.get("modifier")}) has no theta to be analysed at. Give one: '
                    f'"result_from" a calibration that fitted it, or a point set step with '
                    f'{{"entry": "{mod["name"]}", "theta": ...}}.')
            values.append(float(theta))
        return names, values

    def _run_uq(self, step, settings):
        from libcuflynx.parsers.PrimitiveParsers import ObsAndParamDataParser
        from libcuflynx.scripts.param_id_run_script import run_uq

        generated, params_for_id, _, fixed, _ = self.on_root(
            lambda: self._prepare(step, settings))
        info = ObsAndParamDataParser().get_param_id_info(params_for_id)
        # the point is the result_from step's values, which the model was generated with, or
        # the model's own
        names, at = self.on_root(lambda: self._model_point(step, generated, params_for_id))
        method = uq_method(settings)
        out_dir = os.path.join(self.step_dir(step.id), UQ_DIR)
        inp = self._inp(step, settings, generated, params_for_id)
        inp['param_id_output_dir'] = out_dir
        outcome = self._run_stage(step, f'{method} uncertainty quantification', run_uq, inp,
                                  at, method)
        drawn = {}  # rank 0: the Laplace approximation's stored samples

        def finish():
            parameters = parameters_from_info(info['param_names'], info['param_mins'],
                                              info['param_maxs'])
            source = {'workflow': self.workflow.name, 'step': step.id, 'method': method,
                      'model': step.model.as_dict(), 'at': [float(v) for v in at],
                      'obs_data_sha256': sha256(step.obs_data), 'created': _now()}
            record = {'method': method, 'parameters': names, 'at': [float(v) for v in at],
                      'at_source': step.result_from or 'model', 'uq_output_dir': out_dir,
                      'fixed': [dict(row) for _, row in sorted(fixed.items())],
                      'params_for_id_used': params_for_id, 'settings': settings,
                      'generated': generated, 'model_path': generated['model_path']}
            if method == 'mcmc':
                record['distribution_dir'] = self._store_distribution(
                    step, outcome['uq_output_dir'], info, settings)
                return record
            samples, dropped = laplace_samples(outcome['mean'], outcome['covariance'],
                                               parameters, _num_samples(settings))
            if samples.shape[0] < 2:
                raise WorkflowStepError(step.id, f'the Laplace approximation at {list(at)} has '
                                                 f'almost no mass inside the params_for_id '
                                                 f'bounds; it cannot be stored.')
            cov = np.atleast_2d(outcome['covariance'])
            record['laplace'] = {'mean': np.atleast_1d(outcome['mean']).tolist(),
                                 'covariance': cov.tolist(),
                                 'std': np.sqrt(np.diag(cov)).tolist(),
                                 'samples_outside_bounds': dropped}
            distribution = StoredDistribution(parameters, samples, source=dict(
                source, laplace_samples_outside_bounds=dropped))
            record['distribution_dir'] = distribution.save(
                os.path.join(self.step_dir(step.id), DISTRIBUTION_DIR))
            drawn['samples'] = samples
            return record

        record = self.on_root(finish)
        options = _uq_options(settings)
        if not options.get('posterior_predictive', True):
            return record
        checked = self._posterior_predictive(step, options, outcome, drawn.get('samples'))
        return self.on_root(lambda: dict(record, **checked))

    def _posterior_predictive(self, step, options, outcome, samples):
        '''The posterior predictive check of a uq step (``param_id.posterior_predictive``), on
        every rank: draws from the MCMC chain (after ``burn_in``) or from the Laplace
        approximation's ``samples`` (rank 0), simulated by the step's engine. Rank 0 saves it in
        ``<step dir>/posterior_predictive`` and returns the result keys. A check that fails is
        a warning and a ``posterior_predictive_summary.error``: the posterior is the step's
        result.'''
        from libcuflynx.param_id import posterior_predictive as pp
        directory = os.path.join(self.step_dir(step.id), POSTERIOR_PREDICTIVE_DIR)
        try:
            result = pp.posterior_predictive(
                client=outcome['engine'],
                num_samples=int(options.get('posterior_predictive_samples',
                                            POSTERIOR_PREDICTIVE_SAMPLES)),
                burn_in=float(options.get('burn_in', 0.5)),
                output_dir=outcome['uq_output_dir'] if samples is None else None,
                samples=samples, save=False)
        except Exception as exc:
            def failed():
                warnings.warn(f'step "{step.id}": the posterior predictive check failed: {exc}',
                              UserWarning, stacklevel=2)
                return {'posterior_predictive_summary': {'error': str(exc)}}
            return self.on_root(failed)

        def save():
            samples_path, coverage_path = result.save(directory)
            series = os.path.join(directory, pp.SERIES_FILE)
            levels = {level: {key: row[key] for key in ('predictive_coverage',
                                                         'sample_interval_coverage')}
                      for level, row in result.coverage['levels'].items()}
            return {'posterior_predictive_dir': directory,
                    'posterior_predictive_draws': samples_path,
                    'posterior_predictive_coverage': coverage_path,
                    'posterior_predictive_series': series if os.path.isfile(series) else None,
                    'posterior_predictive_summary': {
                        'num_samples': int(result.predictions.shape[0]),
                        'samples_that_failed_to_simulate': int(result.failures),
                        'drawn_from': result.chain_info.get('source', 'chain'),
                        'levels': levels}}

        return self.on_root(save)

    def _run_set(self, step, settings):
        return self.on_root(lambda: {'set': self._declared_values(step),
                                     'thetas': [dict(entry) for entry in step.set_thetas]})

    def _target_values(self, step):
        '''``{model name: row}`` of the target as it stands, generated under the step's
        directory.'''
        try:
            generated = generate_model(
                self.workflow.target, os.path.join(self.step_dir(step.id), 'model'), step.id,
                library_inputs=self.resolved.library_inputs,
                model_type=self.workflow.settings.get('model_type', STEP_DEFAULTS['model_type']),
                solver=self.workflow.settings.get('solver'))
        except UnsetModelParameters as exc:
            raise WorkflowStepError(step.id, f'the target leaves {exc.model_names} without a '
                                             f'value, so it has no value to read here. Scale a '
                                             f'step\'s value, and give "units".') from exc
        return {r['variable_name']: r for r in read_parameter_rows(generated['parameters_path'])}

    def _declared_values(self, step):
        '''The records of a set step's entries, in the shape of a calibrated value.'''
        resolved = self.resolved.steps[step.id]
        step_dir = self.step_dir(step.id)
        if os.path.isdir(step_dir):
            shutil.rmtree(step_dir)
        os.makedirs(step_dir)
        from_steps, target = {}, None
        records = []
        for entry, parameter in zip(step.set_values, resolved.parameters):
            name = resolved.target_model_name(parameter)
            source, base, units = entry.get('of'), None, entry.get('units')
            if source not in (None, INHERITED_BASE):
                if source not in from_steps:
                    from_steps[source] = self.fixed_values(step, [source])
                row = from_steps[source].get(name)
                if row is None:
                    raise WorkflowStepError(step.id, f'step "{source}" has no value of '
                                                     f'"{entry["parameter"]}" to scale.')
                base, units = row['value'], units or row.get('units', '')
            elif source == INHERITED_BASE or units is None:
                if target is None:
                    target = self._target_values(step)
                if name not in target:
                    raise WorkflowStepError(step.id, f'the target has no parameter '
                                                     f'"{entry["parameter"]}".')
                units = units or target[name].get('units', '')
                if source == INHERITED_BASE:
                    base = float(target[name]['value'])
            value = entry['value'] if source is None else entry['scale'] * base
            records.append({
                'module': parameter.vessel, 'param': parameter.param, 'model_name': name,
                'target_model_name': name,
                'target_host_name': naming.host_row_name(parameter.vessel, parameter.param),
                'target_parameterisation_name': resolved.target_parameterisation_name(parameter),
                'value': float(value), 'units': units, 'origin': SET_BY_HAND,
                'reason': entry['reason'],
                **({'scale': entry['scale'], 'of': source, 'base': float(base)}
                   if source is not None else {})})
        return records

    def _run_combine(self, step, settings):
        return self.on_root(lambda: self._combined_values(step))

    def _combined_values(self, step):
        '''The result of a combine step: one record per parameter its inputs give, in the shape
        of a calibrated value named as the target names it, with ``combined_from`` (the inputs
        that gave the parameter) and, for ``best_fit``, ``chosen_from``.'''
        step_dir = self.step_dir(step.id)
        if os.path.isdir(step_dir):
            shutil.rmtree(step_dir)
        os.makedirs(step_dir)
        method = step.combine['method']
        given = {}      # {target model name: [(input id, record)]}
        results = {}
        for source in step.combine_from:
            results[source] = self.require_current(source, step.id)
            for record in given_values(results[source]):
                given.setdefault(record['target_model_name'], []).append((source, record))
        out = {'method': method, 'combine_from': list(step.combine_from)}
        if method == 'best_fit':
            partly = {name: [s for s, _ in rows] for name, rows in sorted(given.items())
                      if len(rows) != len(step.combine_from)}
            if partly:
                raise WorkflowStepError(
                    step.id, f'it takes the best fit of {step.combine_from}, which must '
                             f'calibrate the same parameters, but only some of them calibrated '
                             f'{partly}.')
            costs = {source: results[source].get('best_cost') for source in step.combine_from}
            missing = sorted(source for source, cost in costs.items() if cost is None)
            if missing:
                raise WorkflowStepError(step.id, f'{missing} recorded no cost to compare.')
            chosen = min(step.combine_from, key=lambda source: costs[source])
            out.update(chosen_from=chosen, best_cost=costs[chosen], costs=costs)
        records = []
        for name, rows in sorted(given.items()):
            source, record = next((row for row in rows if row[0] == out.get('chosen_from')),
                                  rows[0])
            units = sorted({r.get('units', '') for _, r in rows})
            if len(units) > 1:
                raise WorkflowStepError(step.id, f'its inputs give "{name}" in different '
                                                 f'units: {units}.')
            value = record['value'] if method == 'best_fit' else \
                float(np.mean([r['value'] for _, r in rows]))
            vessel = self.resolved.steps[source].to_target(record['module'])
            combined = {
                'module': vessel, 'param': record['param'], 'model_name': name,
                'target_model_name': name, 'target_host_name': record['target_host_name'],
                'target_parameterisation_name': record['target_parameterisation_name'],
                'value': float(value), 'units': units[0],
                'combined_from': [s for s, _ in rows]}
            if method == 'best_fit':
                combined['chosen_from'] = source
            elif all(r.get('origin') == SET_BY_HAND for _, r in rows):
                combined.update(origin=SET_BY_HAND, reason='the mean of the values set by hand '
                                f'in steps {", ".join(s for s, _ in rows)}')
            records.append(combined)
        return dict(out, calibrated=records)

    def _run_sweep(self, step, settings):
        resolved = self.resolved.steps[step.id]
        where = f'{step.kind} step "{step.id}"'
        settings = dict({'model_type': STEP_DEFAULTS['model_type']}, **settings)
        generated, _, units, fixed, _ = self.on_root(lambda: self._prepare(step, settings))
        if resolved.model.kind == 'external':
            # the sweep builds its helper from solver_info, without the user_inputs parser
            external = external_inputs(settings, generated)
            settings = dict(settings, model_type=external['model_type'], solver=external['solver'],
                            solver_info=dict(external['solver_info'], external_parameters_file=
                                             external['external_parameters_file']))

        def at_the_model():
            rows = {r['variable_name']: r for r in read_parameter_rows(
                generated['parameters_path'])}
            nominal, swept_units = {}, {}
            for name, parameter in zip(step.sweep['parameters'], resolved.parameters):
                if parameter.model_name not in rows:
                    raise WorkflowError(f'{where}: its model has no parameter "{name}" '
                                        f'({generated["parameters_path"]}).')
                nominal[name] = float(rows[parameter.model_name]['value'])
                swept_units[name] = units.get(parameter.model_name, '')
            return nominal, swept_units

        nominal, swept_units = self.on_root(at_the_model)
        names, points, by_point, solver_info = self._run_stage(
            step, 'sweep', sweep_runs.run_points, step.sweep, generated, settings,
            self.obs_data_read(step), nominal, self.comm, where)

        def finish():
            record = sweep_runs.write_results(
                step.sweep, names, points, by_point, nominal, swept_units,
                os.path.join(self.step_dir(step.id), sweep_runs.SWEEP_DIR), where)
            return dict(record, fixed=[dict(row) for _, row in sorted(fixed.items())],
                        settings=settings, solver_info=solver_info, generated=generated,
                        model_path=generated['model_path'])

        return self.on_root(finish)

    def figure_due(self, step):
        '''Rank 0: whether a figure step is to be drawn: every step it reads has a current
        result, and it has none itself, or its script, files or declaration changed, or a step
        it reads ran again after it.'''
        try:
            sources = [self.require_current(source, step.id) for source in step.depends_on]
        except WorkflowError:
            return False
        result = self.step_result(step.id)
        if result is None or result.get('status') != 'done' or \
                self.changed_inputs(step.id, result):
            return True
        return any((source.get('finished') or '') > (result.get('started') or '')
                   for source in sources)

    def _run_figure(self, step, settings):
        def draw():
            for source in step.depends_on:
                self.require_current(source, step.id)
            step_dir = self.step_dir(step.id)
            if os.path.isdir(step_dir):
                shutil.rmtree(step_dir)
            os.makedirs(step_dir)
            out_dir = os.path.join(step_dir, figures.FIGURE_DIR)
            try:
                paths = figures.run_figure(self.output_dir, step, out_dir)
            except WorkflowError:
                raise
            except Exception as exc:
                raise WorkflowStepError(
                    step.id, f'its function raised {type(exc).__name__}: {exc}') from exc
            declared = figures.declaration(step)
            return {'function': declared['function'], 'figure_inputs': declared['inputs'],
                    'figure_dir': out_dir, 'figures': paths}

        return self.on_root(draw)

    def _store_distribution(self, step, uq_dir, info, settings):
        chain_path = os.path.join(uq_dir, 'mcmc_chain.npy')
        if not os.path.isfile(chain_path):
            raise WorkflowStepError(step.id, f'store_distribution is set but MCMC wrote no '
                                             f'chain ({chain_path}).')
        chain = np.load(chain_path)
        start = burn_in_index(settings, chain.shape[0])
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
                'write_cuflynx_study': self._writes_study(),
                'cuflynx_study_copy_inputs':
                    self.workflow.settings.get('cuflynx_study_copy_inputs', True),
                'cuflynx_working_dir': self.output_dir}

    def packed_inputs(self):
        '''Rank 0: ``{real path of an input: its copy}`` -- the workflow packed (pack_workflow)
        into ``<output_dir>/inputs/workflows/<name>/<first used>/`` (``cuflynx_study._place``:
        one dated folder per content, so a run of an unchanged workflow reuses its folder), so
        the working dir holds every input it was run from. Empty when the study does not copy
        its inputs, or the workflow has no file to pack (a failure to pack is a warning).'''
        if getattr(self, '_packed', None) is not None:
            return self._packed
        self._packed = {}
        if not cuflynx_study.copies_inputs(self._study_inp(self.output_dir, self.workflow.name)) \
                or not self.workflow.path:
            return self._packed
        from libcuflynx.calibration_workflow.pack import pack_workflow
        inputs = os.path.join(self.output_dir, cuflynx_study.INPUTS_DIR)
        os.makedirs(inputs, exist_ok=True)
        staging = tempfile.mkdtemp(prefix='.packing_', dir=inputs)
        try:
            packed = pack_workflow(self.workflow.path, staging, self.workflow.module_library_dirs)
            files = {os.path.relpath(os.path.join(root, name), staging): os.path.join(root, name)
                     for root, _, names in os.walk(staging) for name in names}
            dest = cuflynx_study._place(self.output_dir,
                                        os.path.join('workflows', self.workflow.name), files,
                                        WORKFLOW_FILE_NAME)
        except Exception as error:
            print(f'[cuflynx_study] WARNING: could not pack the workflow into {inputs}: {error}')
            return self._packed
        finally:
            shutil.rmtree(staging, ignore_errors=True)
        self._packed = {real: os.path.join(dest, relative)
                        for real, relative in packed['sources'].items()}
        self._packed[os.path.realpath(self.workflow.path)] = os.path.join(dest, WORKFLOW_FILE_NAME)
        return self._packed

    def _as_packed(self, value):
        '''A path, or each path in a model's dict, as its packed copy when it has one.'''
        if isinstance(value, dict):
            return {k: self._as_packed(v)
                    if k in ('module_array', 'parameters', 'external_model_path',
                             'cellml_model_path') else v
                    for k, v in value.items()}
        if isinstance(value, str) and value:
            return self.packed_inputs().get(os.path.realpath(value), value)
        return value

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
                'id': step.id, 'kind': step.kind,
                'model': self._as_packed(_model_files(step.model)),
                'obs_data': self._as_packed(rs.obs_data_path),
                'params_for_id': self._as_packed(step.params_for_id),
                'place': rs.place if step.model is not None else '',
                'status': 'failed' if step.id == failed else ('done' if done else 'not_run'),
                'run': (result or {}).get('param_id_output_dir') if done else None,
                'outputs': {k: (result or {}).get(k) for k in STEP_OUTPUTS[step.kind]}
                if done else None,
                # only a calibration writes one; a set step, say, has none to name
                'manifest': self.step_manifest(step.id)
                if done and os.path.isfile(self.step_manifest(step.id)) else None}
            entry.update({key: ids for key, ids in step.sources().items() if ids or
                          key in ('fixed_from', 'priors_from')})
            if step.kind == 'reduction':
                entry['reduction'] = dict(step.reduction)
            steps.append(entry)
        target = (workflow_result or {}).get('target_model') or {}
        block = {'name': self.workflow.name,
                 'file': os.path.join(self.output_dir, WORKFLOW_FILE_NAME),
                 'source': self._as_packed(self.workflow.path),
                 'result': result_path if os.path.isfile(result_path) else None,
                 'complete': all(s['status'] == 'done' for s in steps),
                 'target': {'model': self._as_packed(_model_files(self.workflow.target)),
                            'calibrated_parameters':
                                (workflow_result or {}).get('calibrated_parameters_file')},
                 'steps': steps}
        if self.resolved.target.kind == 'external':
            # an external model is not generated: the target is its .py run with the merged
            # values in this parameters file
            block['target']['generated_parameters'] = target.get('parameters_path')
        else:
            block['target']['generated_model'] = target.get('model_path')
        inp = self._study_inp(self.output_dir, self.workflow.name)
        if target.get('model_path'):
            inp['model_path'] = target['model_path']
        manifest = cuflynx_study.update_study(inp, workflow=block)
        for entry in steps:
            if manifest and entry['manifest'] and os.path.isfile(entry['manifest']):
                cuflynx_study.update_study(
                    self._study_inp(os.path.dirname(entry['manifest']), entry['id']),
                    workflow={'manifest': manifest, 'step': entry['id']})

    def write_provenance(self):
        '''Write the run's provenance graph and diagram. A failure is a warning, as for any
        manifest.'''
        from libcuflynx.reporting.provenance import write_provenance
        try:
            write_provenance(self.output_dir)
        except Exception as exc:
            warnings.warn(f'the provenance of {self.output_dir} was not written: {exc}',
                          UserWarning, stacklevel=2)

    def merge(self, order, generate_target=True, write_calibrated=False):
        results = {s.id: self.step_result(s.id) for s in self.workflow.steps}
        merged = {}
        claimed = {}
        for step in order:
            result = results.get(step.id)
            if not result or result.get('status') != 'done':
                continue
            for record in target_values(step, result):
                claimed.setdefault(record['target_model_name'], []).append(step.id)
                merged[record['target_model_name']] = {
                    'model_name': record['target_model_name'],
                    'host_name': record['target_host_name'],
                    'parameterisation_name': record['target_parameterisation_name'],
                    'value': record['value'], 'units': record.get('units', ''),
                    'from_step': step.id, **_how_set(record)}
        check_merged_once(self.workflow, claimed, on_clash=self.target_overlap)
        complete = all(r and r.get('status') == 'done' for r in results.values())
        # a figure passes no value on, so the calibrated parameters do not wait for one
        values_complete = all(r and r.get('status') == 'done' for sid, r in results.items()
                              if self.workflow.step(sid).kind != 'figure')
        target = self.resolved.target
        workflow_result = {
            'workflow_name': self.workflow.name,
            'workflow_file': self.workflow.path,
            'workflow_sha256': workflow_digest(self.workflow.path) if self.workflow.path else None,
            'target': self.workflow.target.as_dict(),
            'order': [s.id for s in order],
            'complete': complete,
            'finished': _now(),
            'libcuflynx_version': _libcuflynx_version(),
            'libcuflynx_git': git_state(repo_root()),
            'project_git': git_state(os.path.dirname(self.workflow.path or '')),
            'libcellml_version': _libcellml_version(),
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
                                            f'step {r["from_step"]}', **_how_set(r)}
                         for r in workflow_result['merged_parameters']]
            target_dir = os.path.join(self.output_dir, TARGET_DIR)
            if os.path.isdir(target_dir):
                shutil.rmtree(target_dir)
            generated = generate_model(
                self.workflow.target, os.path.join(target_dir, 'model'), TARGET_PREFIX, overrides,
                library_inputs=self.resolved.library_inputs,
                model_type=self.workflow.settings.get('model_type', STEP_DEFAULTS['model_type']),
                solver=self.workflow.settings.get('solver'))
            names = {r['variable_name'] for r in read_parameter_rows(generated['parameters_path'])}
            missing = sorted(r['model_name'] for r in workflow_result['merged_parameters']
                             if r['model_name'] not in names)
            if missing:
                raise WorkflowError(
                    f'the target {self.workflow.target.label()} has no parameter {missing}, '
                    f'which the steps calibrated. Name where each step\'s model sits in the '
                    f'target: "submodule" or "module" for a library module, "module_map" for a '
                    f'module array.')
            workflow_result['target_model'] = {
                'model_path': generated['model_path'],
                'parameters_path': generated['parameters_path']}
        if write_calibrated and merged:
            if not values_complete:
                raise WorkflowError('the calibrated parameters file is written only for a '
                                    'complete workflow run; some steps have no result.')
            generated_path = (workflow_result['target_model'] or {}).get('parameters_path')
            workflow_result['calibrated_parameters_file'] = write_calibrated_parameters(
                target, workflow_result['merged_parameters'], self.workflow.name,
                library_inputs=self.resolved.library_inputs, generated_path=generated_path)
            if target.kind == 'cellml':
                workflow_result['calibrated_model_file'] = write_calibrated_cellml(
                    target, workflow_result['calibrated_parameters_file'])
        _write_json(os.path.join(self.output_dir, WORKFLOW_RESULT), workflow_result)
        return workflow_result


#: the model type and solver an external model runs with, whatever the settings say
EXTERNAL_MODEL_TYPE, EXTERNAL_SOLVER = 'external_python', 'external'


def external_inputs(settings, generated):
    '''The user_inputs that run an external model: its model type, solver, file and the values
    file generate_external wrote. ``solver_info`` keeps the settings' ``user_config``.'''
    solver_info = dict(settings.get('solver_info') or {},
                       solver=EXTERNAL_SOLVER, method=EXTERNAL_SOLVER)
    return {'model_type': EXTERNAL_MODEL_TYPE, 'solver': EXTERNAL_SOLVER,
            'solver_info': solver_info, 'external_model_path': generated['model_path'],
            'external_parameters_file': generated['parameters_path']}


def write_calibrated_cellml(target, calibrated_parameters, path=None):
    '''``path`` (default ``<stem>_calibrated.cellml`` beside the CellML target): the target's
    CellML (flattened when it has imports) with every value of ``calibrated_parameters`` (a
    calibrated parameters file of it) that differs from its own written in. Returns ``path``.'''
    from libcuflynx.calibration_workflow.cellml_files import (read_cellml_constants,
                                                              write_cellml_with_values)
    from libcuflynx.param_id.calibrated_model import calibrated_model_name
    from libcuflynx.utilities import parameters_file as pf
    own = read_cellml_constants(target.model.path, target.model.label())
    values = {}
    for place, leaf in pf.leaves(pf.read_parameters_file(calibrated_parameters, kind=pf.MODEL)):
        name = '/'.join(place)
        if name in own and float(leaf['value']) != own[name]['value']:
            values[name] = float(leaf['value'])
    path = path or os.path.join(os.path.dirname(target.model.path),
                                calibrated_model_name(target.model.prefix))
    return write_cellml_with_values(target.model.path, values, path, target.model.label())


def laplace_samples(mean, covariance, parameters, n, seed=0):
    '''``(samples, dropped)``: ``n`` draws of the Laplace approximation N(``mean``,
    ``covariance``), those outside the parameters' bounds dropped (``dropped`` of them), as a
    stored distribution's samples.'''
    rng = np.random.default_rng(seed)
    draws = rng.multivariate_normal(np.atleast_1d(mean), np.atleast_2d(covariance), size=n)
    lo = np.array([p.min for p in parameters])
    hi = np.array([p.max for p in parameters])
    inside = np.all((draws > lo) & (draws < hi), axis=1)
    return draws[inside], int(n - inside.sum())


def _num_samples(settings):
    return int(_uq_options(settings).get('num_samples', LAPLACE_SAMPLES))


def under(base_rows, overrides):
    '''``base_rows`` without those ``overrides`` set again, then ``overrides``: the rows a step
    model is generated with.'''
    replaced = {naming.host_model_name(r['variable_name']) for r in overrides}
    return [r for r in base_rows
            if naming.host_model_name(r['variable_name']) not in replaced] + list(overrides)


def generate_step_model(resolved_step, work_dir, prefix, rows, obs_data, library_inputs,
                        settings):
    '''Generate the model of ``resolved_step`` with parameter ``rows``; the step's protocol
    (``obs_data``, if it has one) gives the parameters nothing else does. Parameters still
    without a value are an UnsetParametersError naming them as the target's parameters file
    does.'''
    try:
        return generate_model(
            resolved_step.model.model, work_dir, prefix, rows, library_inputs=library_inputs,
            model_type=settings.get('model_type', STEP_DEFAULTS['model_type']),
            solver=settings.get('solver'),
            protocol=protocol_values(obs_data) if obs_data else None,
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


def burn_in_index(settings, num_steps):
    '''The first step of a chain of ``num_steps`` after the burn-in the step's UQ options give
    (``burn_in``: a fraction below 1, else a step count; 0.5 by default).'''
    burn_in = float(_uq_options(settings).get('burn_in', 0.5))
    return int(num_steps * burn_in) if burn_in < 1 else int(burn_in)


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
    result, or a step has not run or is stale (its inputs changed since it ran).

    The values the target's parameters file does not set are those the run used: the
    generated ``<prefix>_parameters.json`` of the target model it generated in ``output_dir``
    (``_target/``), whatever the module library holds now. Only when the run has none is the
    target generated again, against the library as it is now.'''
    workflow = source if hasattr(source, 'steps') else load_workflow(source, module_library_dirs)
    result_path = os.path.join(output_dir, WORKFLOW_RESULT)
    if not os.path.isfile(result_path):
        raise WorkflowError(f'{output_dir} holds no run of workflow "{workflow.name}" '
                            f'({WORKFLOW_RESULT} is missing).')
    status = workflow_status(workflow, output_dir)
    waiting = [s for s, v in status['steps'].items()
               if (v['status'] != 'done' or v['stale']) and workflow.step(s).kind != 'figure']
    if waiting:
        raise WorkflowError(f'workflow "{workflow.name}": steps {waiting} have no current result '
                            f'in {output_dir} (not run, or their inputs changed since); run '
                            f'them before writing the calibrated parameters.')
    result = _read_json(result_path)
    resolved = resolve_workflow(workflow, target_overlap='error')
    return write_calibrated_parameters(resolved.target, result['merged_parameters'],
                                       workflow.name, path=path,
                                       library_inputs=resolved.library_inputs,
                                       generated_path=run_target_parameters(output_dir, result))


def run_target_parameters(output_dir, workflow_result=None):
    '''The generated ``<prefix>_parameters.json`` of the target model the run in ``output_dir``
    generated (``workflow_result['target_model']['parameters_path']``, else its place under
    ``_target/``); None when the run has none. WorkflowError when the run recorded a parameters
    CSV, which libcuflynx no longer reads.'''
    from libcuflynx.utilities.parameters_file import EXTENSION, parameters_file_name
    target_dir = os.path.join(output_dir, TARGET_DIR)
    recorded = ((workflow_result or {}).get('target_model') or {}).get('parameters_path')
    if recorded and os.path.splitext(recorded)[1].lower() != EXTENSION:
        raise WorkflowError(
            f'the run in {output_dir} records its target model\'s parameters as {recorded}, '
            f'which libcuflynx no longer reads (a generated model\'s parameters are '
            f'{parameters_file_name("<prefix>")}). Run the workflow again, or remove '
            f'{target_dir} to have the target generated again against the module library as it '
            f'is now.')
    candidates = [recorded, os.path.join(target_dir, 'model', 'generated_models', TARGET_PREFIX,
                                         parameters_file_name(TARGET_PREFIX))]
    return next((p for p in candidates if p and os.path.isfile(p)), None)


def target_generated_parameters(target, library_inputs=None, generated_path=None):
    '''Every value the target's model uses, as a parameters document: the generated parameters
    file (``<prefix>_parameters.json`` beside a generated model) at ``generated_path``, or,
    without one, that of the target generated as it stands in a temporary directory. A module
    array's is a model's document; a library module's is rooted at the module, as its
    parameterisations are (its instance's values and the model's globals). WorkflowError when,
    generated again, the target alone leaves a parameter without a value.'''
    from libcuflynx.utilities import parameters_file as pf
    if generated_path is None:
        with tempfile.TemporaryDirectory() as work_dir:
            try:
                generated = generate_model(target.model, os.path.join(work_dir, 'model'),
                                           TARGET_PREFIX, library_inputs=library_inputs)
            except UnsetModelParameters as exc:
                raise WorkflowError(f'the target leaves {exc.model_names} without a value, so '
                                    f'its calibrated parameters cannot hold every value its '
                                    f'model uses; set them in its parameters file.') from exc
            doc = pf.read_parameters_file(generated['parameters_path'])
    else:
        doc = pf.read_parameters_file(generated_path)
    if target.kind in ('array', 'external', 'cellml'):
        return doc
    module = pf.new_document()
    if pf.GLOBAL in doc:
        module[pf.GLOBAL] = doc[pf.GLOBAL]
    module.update(doc.get(naming.STEP_VESSEL) or {})
    return module


def write_calibrated_parameters(target, merged_parameters, workflow_name, path=None,
                                library_inputs=None, generated_path=None):
    '''``path`` (default ``<name>_calibrated_parameters.json`` beside the target's parameters
    file, ``<name>`` its parameterisation, or its module array's prefix): the target's parameters
    file (nested JSON, utilities/parameters_file.py), completed with every other value its model
    uses (:func:`target_generated_parameters`; each one's data_reference names the
    parameterisation it came from, ``from <parameterisation>: <reference>``), with each merged
    value set at the path the target names it by -- ``<vessel>/<variable>`` in a
    module array's file (a vessel inside a super instance, ``sys__C/q``, at ``sys/C/q``), the
    module's own path in a parameterisation. So the result reproduces the run without the
    module library. Only ``path`` is written; the target's files are not changed.

    A merged value's leaf loses ``derivation`` and ``reason`` and gets origin ``calibrated``
    (``parameters_file.calibrated_leaf``), or ``set_by_hand`` with the set step's reason, so
    ``cuflynx-derive --write`` never overwrites it. The derived value is not lost: the target's
    parameters file still holds it, its derivations file still calculates it
    (``cuflynx-derive``), and the step's generated parameters file (``generated.parameters_path``
    in its result) holds it as the value the step started from.'''
    from libcuflynx.utilities import parameters_file as pf
    as_model = target.kind in ('array', 'external', 'cellml')
    kind = pf.MODEL if as_model else pf.MODULE
    doc = target_generated_parameters(target, library_inputs, generated_path)
    # the target's own leaves as it writes them (a library's "sourced" kept); an external
    # model's are its class defaults, already in doc
    if target.parameters_path:
        for place, leaf in pf.leaves(pf.read_parameters_file(target.parameters_path, kind=kind)):
            pf.set_leaf(doc, place, leaf)
    today = datetime.date.today().isoformat()
    for record in merged_parameters:
        name = record['host_name'] if as_model else record['parameterisation_name']
        place = pf.name_path(name, kind)
        how = 'set by hand in' if record.get('origin') else 'calibrated by'
        leaf = pf.calibrated_leaf(pf.get_leaf(doc, place) or {'units': record.get('units', '')},
                                  repr(float(record['value'])),
                                  f'{how} workflow {workflow_name}, step '
                                  f'{record["from_step"]} ({today})')
        # a set step's value: origin set_by_hand, with its reason
        leaf.update(_how_set(record))
        if 'sourced' in leaf:
            leaf['sourced'] = 'no'
        pf.set_leaf(doc, place, leaf)
    return pf.write_parameters_file(path or target.file('calibrated_parameters.json'), doc)


def _write_params_for_id_with_thetas(source_path, out_path, thetas, step_id):
    '''The step's own params_for_id with ``theta`` on each modifier entry named in ``thetas``
    (``{entry name: theta}``): where the entry starts and what it is analysed about.'''
    from libcuflynx.utilities import params_for_id_file as pfi
    doc = pfi.read_params_for_id(source_path, f'step "{step_id}": ', keep_text=True)
    entries = [dict(entry, theta=thetas[entry['name']])
               if entry.get('modifies') and entry.get('name') in thetas else entry
               for entry in doc.get('params') or []]
    pfi.write_params_for_id(out_path, dict(doc, params=entries))


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
    files, where its values land in the target, and what it is given by earlier steps.
    ``target_overlap`` is the message a run of every step would refuse the workflow with when
    steps that do not take from each other give the target one parameter (also a UserWarning
    here), else None.'''
    workflow = source if hasattr(source, 'steps') else load_workflow(source, module_library_dirs)
    resolved = resolve_workflow(workflow)
    steps = []
    for step in workflow.topological_order():
        rs = resolved.steps[step.id]
        fixed = []
        for source_id in step.fixed_from:
            vessel_of = resolved.map_between(source_id, step.id)
            subset = step.fixed_parameters.get(source_id)
            for p in resolved.steps[source_id].parameters:
                if subset is not None and p.model_name not in subset:
                    continue
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
        if step.kind == 'combine':
            entry['combine_from'] = list(step.combine_from)
            entry['combine'] = dict(step.combine, gives=[
                {'target_model_name': rs.target_model_name(p),
                 'target_parameterisation_name': rs.target_parameterisation_name(p)}
                for p in rs.parameters])
        for key in ('emulator_from', 'emulator_dir', 'result_from', 'start_from',
                    'sensitivity_from', 'params_for_id_from'):
            if getattr(step, key):
                entry[key] = getattr(step, key)
        if rs.selection is not None:
            # what the selection runs as: its library module at one instance, or the cut array
            entry['selection'] = (
                {'runs_as': 'library_module', 'module': rs.selection.model.as_dict(),
                 'place': rs.selection.vessel} if rs.selection.vessel else
                {'runs_as': 'module_array',
                 'records': [r['name'] for r in rs.selection.model.records]})
        if step.kind == 'reduction':
            entry['reduction'] = dict(step.reduction)
        if step.kind == 'set':
            entry['set'] = [dict(e, target_model_name=rs.target_model_name(p))
                            for e, p in zip(step.set_values, rs.parameters)]
            entry['set'] += [dict(e) for e in step.set_thetas]
        if step.kind in SWEEP_KINDS:
            entry['sweep'] = copy.deepcopy(step.sweep)
        if step.kind == 'figure':
            entry['figure'] = copy.deepcopy(step.figure)
        steps.append(entry)
    return {'workflow_name': workflow.name, 'target': workflow.target.as_dict(),
            'target_parameters': resolved.target.parameters_path,
            'target_values': _target_values(resolved.target, steps),
            'module_library_dirs': workflow.module_library_dirs, 'steps': steps,
            'target_overlap': resolved.target_overlap}


def _target_values(target, steps):
    '''Every parameter of the target (a resolve.Resolved* target) before any step runs, as
    ``plan_workflow`` gives it: ``{name, value, units, origin, data_reference, reason,
    calibrated_by}`` in its parameters file's order, then a CellML or external model's constants
    that file does not set. ``name`` is as the parameters file writes it (``aortic_root/C``,
    ``heart/lv/E``; a library module's ``C``, ``lv/E``, ``global/R``; a CellML model's
    ``component/variable``). ``origin`` is the file's (``parameters_file.ORIGINS``), None where it
    records none. ``calibrated_by`` lists the calibration steps (``steps``, the plan's entries)
    whose values land on it.'''
    from libcuflynx.utilities import parameters_file as pf
    from libcuflynx.utilities.parameter_names import model_name

    calibrated_by = {}
    for step in steps:
        for p in step['calibrates']:
            calibrated_by.setdefault(model_name(p['target_parameterisation_name']), []) \
                .append(step['id'])
    path = target.parameters_path
    rows = (pf.read_parameter_rows(path, kind=pf.MODULE if target.kind == 'library' else pf.MODEL)
            if path and os.path.isfile(path) else [])
    constants = getattr(target, 'constants', None) or {
        name: {'value': value} for name, value in (getattr(target, 'parameters', None) or {}).items()}
    named = {row['variable_name'] for row in rows}
    rows += [{'variable_name': name, 'value': c.get('value'), 'units': c.get('units', '')}
             for name, c in constants.items() if name not in named]
    values = []
    for row in rows:
        try:
            value = float(row.get('value'))
        except (TypeError, ValueError):
            value = row.get('value')
        origin = row.get('origin')
        values.append({'name': row['variable_name'], 'value': value,
                       'units': row.get('units') or '',
                       'origin': origin if origin in pf.ORIGINS else None,
                       'data_reference': row.get('data_reference') or '',
                       'reason': row.get('reason') or '',
                       'calibrated_by': calibrated_by.get(model_name(row['variable_name']), [])})
    return values


def run_calibration_workflow(source, *, output_dir=None, module_library_dirs=None,
                             from_step=None, only=None, stale_only=False, dry_run=False,
                             write_calibrated=False, progress=None, comm=None):
    '''Run the calibration workflow ``source`` (a calibration_workflow.json path or its dict).

    Args:
        output_dir: where the run goes (default ``workflow_output/<workflow_name>`` beside
            the workflow file; under the working directory for a workflow given as a dict).
        module_library_dirs: module libraries, added to the file's own.
        from_step: run this step and every step after it in the order; earlier steps' results
            are reused from ``output_dir`` (an error if missing, or if their obs_data,
            params_for_id or parameters changed since).
        only: run just this step, reusing the results it depends on.
        stale_only: without ``from_step`` and ``only``, run just the steps that have not run
            or are stale (:func:`workflow_status`), keeping the results that are current.
        dry_run: resolve and check everything, run nothing; returns :func:`plan_workflow`.
        write_calibrated: write the merged set as the target parameterisation's
            ``<parameterisation>_calibrated_parameters.json`` (complete runs only).
        progress: ``callable(event_dict)``, called on rank 0 with ``event`` in
            ``workflow_started``, ``step_started``, ``step_finished``, ``step_failed``,
            ``step_skipped`` (with ``has_result``, and ``current`` True for a step
            ``stale_only`` left alone or a figure step that is up to date),
            ``workflow_finished``.
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
    # Steps that do not take from each other giving the target one parameter are refused where
    # the target is merged for use: a run of more than one step, an on_target sweep, the
    # calibrated parameters. A run of one other step goes through with a warning, so a second
    # calibration can be run before the combine step that settles it is added.
    target_overlap = 'warn' if only and not workflow.step(only).on_target \
        and not write_calibrated else 'error'
    resolved = resolve_workflow(workflow, target_overlap=target_overlap)
    output_dir = os.path.abspath(output_dir or default_output_dir(workflow))
    run = _Run(resolved, output_dir, comm, progress)
    run.target_overlap = target_overlap
    order = workflow.topological_order()
    if only:
        to_run = [workflow.step(only)]
    elif from_step:
        ids = [s.id for s in order]
        to_run = order[ids.index(from_step):]
    elif stale_only:
        status = run.on_root(lambda: _steps_status(run))
        to_run = [s for s in order
                  if status[s.id]['status'] != 'done' or status[s.id]['stale']]
    else:
        to_run = order
    figure_steps = [s for s in order if s.kind == 'figure']
    forced = [s for s in figure_steps if s.id == only]
    to_run = [s for s in to_run if s.kind != 'figure']

    def start():
        os.makedirs(output_dir, exist_ok=True)
        snapshot = os.path.join(output_dir, WORKFLOW_FILE_NAME)
        if workflow.path:
            shutil.copyfile(workflow.path, snapshot)
        else:
            _write_json(snapshot, workflow.raw)
    run.on_root(start)
    run.record_study(order)
    run.emit('workflow_started', order=[s.id for s in order],
             running=[s.id for s in to_run + forced], output_dir=output_dir)
    for step in order:
        if step not in to_run and step not in figure_steps:
            has_result = run.on_root(
                lambda: (run.step_result(step.id) or {}).get('status') == 'done')
            run.emit('step_skipped', step=step.id, has_result=has_result,
                     current=bool(stale_only and not from_step and not only))
    for index, step in enumerate(to_run):
        try:
            run.run_step(step, index, len(to_run))
        except Exception as exc:
            run.emit('step_failed', step=step.id, index=index, total=len(to_run),
                     error=str(exc))
            run.record_study(order, failed=step.id)
            raise
        run.record_study(order)
    figure_error = None
    for index, step in enumerate(figure_steps):
        if step not in forced and not run.on_root(lambda step=step: run.figure_due(step)):
            run.emit('step_skipped', step=step.id)
            continue
        try:
            run.run_step(step, index, len(figure_steps))
        except Exception as exc:
            run.emit('step_failed', step=step.id, index=index, total=len(figure_steps),
                     error=str(exc))
            run.record_study(order, failed=step.id)
            figure_error = figure_error or exc
            continue
        run.record_study(order)
    result = run.on_root(lambda: run.merge(order, write_calibrated=write_calibrated))
    run.record_study(order, workflow_result=result)
    run.on_root(run.write_provenance)
    run.emit('workflow_finished', complete=result['complete'], output_dir=output_dir)
    if figure_error is not None:
        raise figure_error
    return result


def workflow_status(source, output_dir, module_library_dirs=None):
    """Where each step of a workflow stands in ``output_dir``, cheaply (no generation).

    Returns ``{'workflow_name', 'order', 'complete', 'steps': {id: {'status', 'stale',
    'library_changed', 'best_cost', 'finished', 'point_estimate'}}}`` with ``status`` ``'done'``
    or ``'not_run'``; ``stale`` is True when the step's obs_data, params_for_id, parameters or
    settings changed since it ran, or a step it takes values from is stale or was rerun after
    it. ``library_changed`` says the module library changed; a library holds far more than a
    model uses, so that alone does not make a step stale.
    """
    workflow = source if hasattr(source, 'steps') else load_workflow(source, module_library_dirs)
    resolved = resolve_workflow(workflow)
    steps = _steps_status(_Run(resolved, output_dir, _get_comm(None), None))
    return {'workflow_name': workflow.name, 'order': [s.id for s in workflow.topological_order()],
            'complete': all(v['status'] == 'done' and not v['stale'] for v in steps.values()),
            'steps': steps}


def _steps_status(run):
    '''The ``steps`` of :func:`workflow_status` for the run ``run``.'''
    workflow = run.workflow
    steps = {}
    for step in workflow.topological_order():
        result = run.step_result(step.id)
        done = bool(result) and result.get('status') == 'done'
        stale = library_changed = False
        if done:
            changed = run.changed_inputs(step.id, result)
            library_changed = LIBRARY_INPUT in changed
            stale = bool(set(changed) - {LIBRARY_INPUT})
            for source_id in step.depends_on:
                upstream = steps.get(source_id, {})
                if upstream.get('stale') or upstream.get('status') != 'done' or \
                        (upstream.get('finished') or '') > (result.get('started') or ''):
                    stale = True
        steps[step.id] = {'status': 'done' if done else 'not_run', 'stale': stale,
                          'library_changed': library_changed,
                          'best_cost': (result or {}).get('best_cost') if done else None,
                          'finished': (result or {}).get('finished') if done else None,
                          'point_estimate': (result or {}).get('point_estimate') if done else None}
    return steps


TARGET_VIEW = 'target'


def workflow_model(source, view, *, output_dir, work_dir, module_library_dirs=None,
                   with_own_results=True, validate=True):
    """Generate the model one tab of a workflow shows, from whatever results exist so far.

    ``view`` is a step id, or ``'target'`` for the workflow's target. A step's model gets
    the values its ``fixed_from`` and ``result_from`` steps calibrated (those that have run
    into ``output_dir``) and, with ``with_own_results``, its own calibrated values once it has
    run. The target
    gets every calibrated value there is, mapped into its naming. Nothing is calibrated, and
    nothing in ``output_dir`` is written but the model, under ``work_dir`` (which may be
    ``<output_dir>/models``, :data:`MODELS_DIR`, a folder no step's output can have).

    ``validate=False`` skips the trial run generation ends with (a Myokit compile, about half
    the time of a view), for a caller that compiles or runs the model itself straight after.
    A model with parameters left unset is still refused.

    Returns ``{view, kind ('step' or 'target'), target, submodule_path, model_path,
    flat_model_path (one self-contained CellML file), parameters_path, obs_data_path, params_for_id_path, fixed, calibrated, waiting_for,
    stale, result, param_id_output_dir}``. ``fixed`` lists the values given by earlier
    steps, but not ``result_from``'s values of the parameters the step analyses; ``calibrated``
    the view's own (for the target, the merged set: each value from the last step that set it,
    with ``origin`` and ``reason`` when it was set by hand); ``waiting_for``
    the steps whose results it should have but that have not run; ``stale`` those whose
    inputs changed since they ran. obs_data / params_for_id are None when the parameterisation has
    none (a target with no step of its own). For the target, ``step_id``, its obs_data,
    params_for_id, ``result`` and ``param_id_output_dir`` are those of the last calibration on the
    target that has run, or of the last one when none has.
    """
    workflow = source if hasattr(source, 'steps') else load_workflow(source, module_library_dirs)
    resolved = resolve_workflow(workflow)
    run = _Run(resolved, output_dir, _get_comm(None), None)
    waiting, stale = [], []
    library_inputs = dict(resolved.library_inputs)
    if not validate:
        library_inputs['validate_generated_model'] = False

    def available(step_id):
        result = run.step_result(step_id)
        if result is None or result.get('status') != 'done':
            waiting.append(step_id)
            return None
        if set(run.changed_inputs(step_id, result)) - {LIBRARY_INPUT}:
            stale.append(step_id)
        return result

    # overrides: {host parameters file name: (record, step id)}
    overrides, fixed, calibrated, base_rows = {}, [], [], []
    if view == TARGET_VIEW:
        kind, model = 'target', workflow.target
        latest = {}
        for step in workflow.topological_order():
            result = available(step.id)
            for record in target_values(step, result) if result else []:
                latest[record['target_model_name']] = (record, step.id)
        overrides = {rec['target_host_name']: (rec, sid) for rec, sid in latest.values()}
        calibrated = [{'model_name': name, 'value': rec['value'], 'units': rec.get('units', ''),
                       'from_step': sid, **_how_set(rec)}
                      for name, (rec, sid) in sorted(latest.items())]
        # the inputs and results shown beside the merged values are those of the last
        # calibration on the target that has run (else the last one), as a later step wins
        # every value the steps share
        on_target = [s for s in workflow.topological_order()
                     if s.model == workflow.target and s.kind == 'calibration']
        ran = [s for s in on_target if (run.step_result(s.id) or {}).get('status') == 'done']
        own_step = (ran or on_target or [None])[-1]
        place = ''
        own_result = run.step_result(own_step.id) if own_step else None
    else:
        step = workflow.step(view)
        if step.model is None:
            raise WorkflowError(f'step "{step.id}" is a {step.kind} step, which has no model.')
        kind, model = 'step', step.model
        rs = resolved.steps[step.id]
        place = rs.place
        base_rows = rs.base_rows
        # as the run gives them (_Run._prepare): result_from's values of the step's own
        # parameters are the point it analyses, in its model but not listed as fixed
        own = set()
        if step.result_from and step.params_for_id:
            own = run.own_parameters(step, step.params_for_id)
        for source_id in step.fixed_from + ([step.result_from] if step.result_from else []):
            result = available(source_id)
            if result is None:
                continue
            vessel_of = resolved.map_between(source_id, step.id)
            records = given_values(result) + result['fixed']
            if source_id in step.fixed_parameters:
                records = [r for r in given_values(result)
                           if r['model_name'] in step.fixed_parameters[source_id]]
            for record in records:
                vessel = vessel_of(record['module'])
                if vessel is None:
                    continue
                name = naming.model_name(vessel, record['param'])
                origin = record.get('from_step', source_id)
                overrides[naming.host_row_name(vessel, record['param'])] = (
                    {'value': record['value'], 'units': record.get('units', '')}, origin)
                if source_id == step.result_from and name in own:
                    continue
                fixed = [f for f in fixed if f['model_name'] != name]
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
                    overrides.setdefault(naming.host_row_name(r['module'], r['param']),
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
                                        library_inputs, settings)
    else:
        generated = generate_model(
            model, view_dir, view, rows, library_inputs=library_inputs,
            model_type=settings.get('model_type', STEP_DEFAULTS['model_type']),
            solver=settings.get('solver'))
    if has_factors(obs_data):
        # resolved against the view's model, as a run resolves them against the step's
        resolved_obs = step_input_paths(view_dir, own_step)[0]
        os.makedirs(os.path.dirname(resolved_obs), exist_ok=True)
        model_rows = {r['variable_name']: r
                      for r in read_parameter_rows(generated['parameters_path'])}
        resolve_factors(obs_data, resolved_obs, model_rows,
                        {v['model_name']: v for v in fixed + calibrated}, {}, f'view "{view}"')
        obs_data = resolved_obs

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
    refuse_older_step_keys(raw, snapshot)
    result_path = os.path.join(output_dir, WORKFLOW_RESULT)
    steps = {}
    for step in raw.get('steps', []):
        path = os.path.join(output_dir, step.get('id', ''), STEP_RESULT)
        steps[step.get('id')] = read_step_result(path) if os.path.isfile(path) else None
    return {'output_dir': output_dir, 'workflow': raw,
            'result': _read_json(result_path) if os.path.isfile(result_path) else None,
            'steps': steps}
