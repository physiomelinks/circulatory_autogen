'''
Calibration workflows: ordered calibrations of module instances (typically a supermodule's
submodules, then the supermodule itself), merged into one supermodule instance's parameters.

The workflow is a ``calibration_workflow.json`` (schema
``libcuflynx/schemas/calibration_workflow.schema.json``), usually kept in the target
supermodule instance's directory. Each step calibrates one instance against that instance's
own ``<instance>_obs_data.json`` and ``<instance>_params_for_id.csv``, through the normal
param_id path; a later step can take earlier steps' calibrated values as fixed values
(``fixed_from``) or their stored posteriors as priors (``priors_from``). Editing an obs_data
file and re-running the workflow redoes everything downstream of it.

Not to be confused with ``cuflynx-sequential-param-id``, the unimplemented idea of staged
fitting of *one* model (fit, drop unidentifiable parameters, refit).

Entry points: :func:`run_calibration_workflow` (and the ``cuflynx-calibration-workflow``
command), :func:`plan_workflow`, :func:`load_workflow`, :func:`load_workflow_run`,
:func:`workflow_model` (the model a GUI tab shows), :class:`StoredDistribution`,
:func:`generate_module_instance`.
'''

from libcuflynx.calibration_workflow.distributions import StoredDistribution
from libcuflynx.calibration_workflow.generate import generate_module_instance
from libcuflynx.calibration_workflow.resolve import resolve_workflow
from libcuflynx.calibration_workflow.runner import (TARGET_VIEW, WorkflowStepError,
                                                    load_workflow_run, plan_workflow,
                                                    run_calibration_workflow, workflow_model,
                                                    workflow_status)
from libcuflynx.calibration_workflow.spec import (ALLOWED_SETTINGS, PRIOR_KINDS,
                                                  WORKFLOW_FILE_NAME, Workflow, WorkflowError,
                                                  load_workflow)

__all__ = ['ALLOWED_SETTINGS', 'PRIOR_KINDS', 'WORKFLOW_FILE_NAME', 'StoredDistribution',
           'Workflow', 'WorkflowError', 'WorkflowStepError', 'generate_module_instance',
           'load_workflow', 'load_workflow_run', 'plan_workflow', 'resolve_workflow',
           'run_calibration_workflow', 'TARGET_VIEW', 'workflow_model',
           'workflow_status']
