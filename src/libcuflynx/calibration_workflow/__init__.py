'''
Calibration workflows: calibrations, and the analyses around them, run in order; the
calibrations' results are merged into one model.

A ``calibration_workflow.json`` (schema ``libcuflynx/schemas/calibration_workflow.schema.json``)
names a ``target`` model and a list of steps. Each step names the three inputs of an ordinary
param_id run -- a ``model``, an ``obs_data`` file and a ``params_for_id`` file -- where a model
is a module array of CA, a module of a module library, or (with a module-array target) a
selection of the target's instances, which runs as one of the other two (``select``). A
step's ``kind`` makes it a
sensitivity analysis, a reduction (the parameters a later calibration fits, chosen by Sobol
indices), an emulator training or an identifiability analysis instead (see ``spec``). A later
step can take earlier steps' calibrated values as fixed values (``fixed_from``) or their stored
posteriors as priors (``priors_from``). Editing an input and re-running the workflow redoes
everything downstream of it.

Entry points: :func:`run_calibration_workflow` (and the ``cuflynx-calibration-workflow``
command), :func:`pack_workflow` (every file a run needs, in one directory),
:func:`plan_workflow`, :func:`load_workflow`, :func:`load_workflow_run`,
:func:`workflow_model` (the model a GUI tab shows), :class:`StoredDistribution`,
:func:`generate_model`.
'''

from libcuflynx.calibration_workflow.distributions import StoredDistribution
from libcuflynx.calibration_workflow.generate import generate_model, generate_parameterisation
from libcuflynx.calibration_workflow.pack import pack_workflow
from libcuflynx.calibration_workflow.resolve import resolve_workflow
from libcuflynx.calibration_workflow.runner import (TARGET_VIEW, WorkflowStepError,
                                                    load_workflow_run, plan_workflow,
                                                    run_calibration_workflow, workflow_model,
                                                    workflow_status,
                                                    write_calibrated_parameters_to)
from libcuflynx.calibration_workflow.select import UnsetParametersError
from libcuflynx.calibration_workflow.spec import (ALLOWED_SETTINGS, PRIOR_KINDS,
                                                  WORKFLOW_FILE_NAME, ModuleArray, Selection,
                                                  Target, Workflow, WorkflowError, load_workflow)

__all__ = ['ALLOWED_SETTINGS', 'PRIOR_KINDS', 'WORKFLOW_FILE_NAME', 'ModuleArray',
           'Selection', 'StoredDistribution', 'Target', 'UnsetParametersError', 'Workflow',
           'WorkflowError', 'WorkflowStepError',
           'generate_model', 'generate_parameterisation',
           'load_workflow', 'load_workflow_run', 'pack_workflow', 'plan_workflow',
           'resolve_workflow',
           'run_calibration_workflow', 'TARGET_VIEW', 'workflow_model',
           'workflow_status', 'write_calibrated_parameters_to']
