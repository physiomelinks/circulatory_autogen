# Calibration workflows

A calibration workflow is an ordered set of steps (calibrations, and the analyses around
them) whose results are merged into one target model; see
[Calibration workflows](../parameter-identification.md#calibration-workflows) for the how-to.
`WorkflowBuilder` makes a workflow with API calls and runs it; the functions below it run,
plan and read a workflow file however it was written.

## Building a workflow

::: libcuflynx.calibration_workflow.builder.WorkflowBuilder
    options:
      members:
        - load
        - name
        - target
        - module_library_dirs
        - output_dir
        - steps
        - update_settings
        - add_calibration
        - add_sensitivity
        - add_reduction
        - add_emulation
        - add_identifiability
        - add_uq
        - add_set
        - add_sweep
        - add_step
        - remove_step
        - add_operation
        - add_cost_func
        - add_modifier_func
        - remove_operation
        - remove_cost_func
        - remove_modifier_func
        - functions
        - funcs_source
        - to_dict
        - check
        - write
        - plan
        - run
        - status
        - result
        - step_dir
        - plot_step
        - model

## What a step is made of

::: libcuflynx.calibration_workflow.builder.library_module

::: libcuflynx.calibration_workflow.builder.module_array

::: libcuflynx.calibration_workflow.builder.external_model

::: libcuflynx.calibration_workflow.builder.cellml_model

::: libcuflynx.calibration_workflow.builder.selection

::: libcuflynx.calibration_workflow.builder.fixed

::: libcuflynx.calibration_workflow.builder.prior

::: libcuflynx.calibration_workflow.builder.set_value

::: libcuflynx.calibration_workflow.builder.scale_value

## Running and reading a workflow file

::: libcuflynx.calibration_workflow.runner.run_calibration_workflow

::: libcuflynx.calibration_workflow.runner.plan_workflow

::: libcuflynx.calibration_workflow.runner.workflow_status

::: libcuflynx.calibration_workflow.runner.workflow_model

::: libcuflynx.calibration_workflow.runner.load_workflow_run

::: libcuflynx.calibration_workflow.pack.pack_workflow

## The functions file

::: libcuflynx.utilities.user_funcs_file
    options:
      members:
        - funcs_file_source
        - write_funcs_file
        - read_funcs_file
        - is_unedited
        - unkept_statements
